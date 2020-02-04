import pytest

from flambe.compile.schema import Schema, Link, CopyLink, MalformedLinkError


class Container:

    def __init__(self, a, b):
        self.a = a
        self.b = b

class Child:

    def __init__(self, val):
        self.val = val


@pytest.fixture
def test_schemas():
    ch1 = Schema(Child, args=[0])
    ch2 = Schema(Child, args=[1])
    con = Schema(Container, kwargs={'a': ch1, 'b': ch2})
    return con, ch1, ch2


class TestSchemaMutate:

    def test_create_schema(self):
        """Test schemas can be created and contain other schemas"""
        ch1 = Schema(Child, args=[0])
        ch2 = Schema(Child, args=[1])
        con = Schema(Container, kwargs={'a': ch1, 'b': ch2})

    def test_dict_fns(self, test_schemas):
        """Test schema values can be accessed, changed via dict fns"""
        con, ch1, ch2 = test_schemas
        assert con['a'] is ch1
        assert con['b'] is ch2
        assert con['a']['val'] == 0
        assert con['b']['val'] == 1
        con['a']['val'] = 2
        con['b']['val'] = 3
        assert con['a']['val'] == 2
        assert con['b']['val'] == 3

    def test_get_param(self, test_schemas):
        """Test schema values can be accessed via get_param"""
        con, ch1, ch2 = test_schemas
        assert con.get_param(('a', 'val')) == 0
        assert con.get_param(('b', 'val')) == 1
        with pytest.raises(KeyError):
            con.get_param(('b', 'v'))
        with pytest.raises(KeyError):
            con.get_param(('c', 'val'))

    def test_set_param(self, test_schemas):
        """Test schema values can be changed via set_param"""
        con, ch1, ch2 = test_schemas
        con.set_param(('a', 'val'), 2)
        con.set_param(('b', 'val'), 3)
        with pytest.raises(KeyError):
            con.set_param(('b', 'v'), 4)
        assert con.get_param(('a', 'val')) == 2
        assert con.get_param(('b', 'val')) == 3
        with pytest.raises(KeyError):
            con.get_param(('b', 'v'))
        with pytest.raises(KeyError):
            con.get_param(('c', 'val'))

    def test_traverse(self, test_schemas):
        """Test traverse iterates over schema in depth first order"""
        con, ch1, ch2 = test_schemas
        expected = {
            tuple(): con,
            ('a',): ch1,
            ('a', 'val'): 0,
            ('b',): ch2,
            ('b', 'val'): 1
        }
        i = 0
        for path, obj in Schema.traverse(con):
            assert obj == expected[path]
            assert list(expected.keys()).index(path) == i
            i += 1

    def test_repr(self, test_schemas):
        """Test schema repr"""
        con, ch1, ch2 = test_schemas
        assert str(con) == ("flambe.compile.schema.Schema(<class 'tests.unit.compile.test_schema."
                            "Container'>, a=flambe.compile.schema.Schema(<class "
                            "'tests.unit.compile.test_schema.Child'>, val=0), "
                            "b=flambe.compile.schema.Schema(<class "
                            "'tests.unit.compile.test_schema.Child'>, val=1))")


class TestSchemaInitialize:

    def test_initialize_creates_instance(self):
        """Test initialization runs for simple schema"""

        class A:
            pass

        schema = Schema(A)
        a = schema.initialize()
        assert isinstance(a, A)

    def test_initialize_captures_kwargs(self):
        """Test initialization captures keyword args"""

        class A:
            def __init__(self, x):
                self.x = x

        kwargs = {'x': 1}
        schema = Schema(A, kwargs=kwargs)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_args(self):
        """Test initialization captures positional args"""

        class A:
            def __init__(self, x):
                self.x = x

        args = [1]
        schema = Schema(A, args=args)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_all_args(self):
        """Test initialization captures positional and keyword args"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        args = [1]
        kwargs = {'y': 2}
        schema = Schema(A, args=args, kwargs=kwargs)
        a = schema.initialize()
        assert a.x == 1
        assert a.y == 2

    def test_initialize_recursive(self):
        """Test initialization will recurse to child schemas"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        class B:
            def __init__(self, z):
                self.z = z
        sb = Schema(B, kwargs={'z': 2})
        sa = Schema(A, args=[1, sb])
        a = sa.initialize()
        assert a.x == 1
        assert a.y.z == 2

    def test_initialize_recursive_deep(self):
        """Test initialization will recurse to grandchild schemas"""
        class A:
            def __init__(self, ax, ay):
                self.ax = ax
                self.ay = ay
        class B:
            def __init__(self, bx, by):
                self.bx = bx
                self.by = by
        class C:
            def __init__(self, cx, cy):
                self.cx = cx
                self.cy = cy
        s1 = Schema(C, args=[None, 1])
        s2 = Schema(B, args=[s1, 2])
        s3 = Schema(A, args=[s2, 3])
        # Check basic usage
        a = s3.initialize()
        assert a.ay == 3
        assert a.ax.by == 2
        assert a.ax.bx.cy == 1
        assert a.ax.bx.cx is None
        # Check container usage
        class c:
            def __init__(self, stage_1):
                self.stage_1 = stage_1
            def get_param(self, path):
                a, b = path[0], path[1:]
                assert a == 'stage_1'
                return self.stage_1.get_param(b)
            def set_param(self, path, value):
                a, b = path[0], path[1:]
                assert a == 'stage_1'
                self.stage_1.set_param(b, value)
        root = c(stage_1=s3)
        cache = {}
        a = s3.initialize(path=('stage_1',), cache=cache, root=root)
        for k in cache:
            assert k[0] == 'stage_1'
        assert a.ay == 3
        assert a.ax.by == 2
        assert a.ax.bx.cy == 1
        assert a.ax.bx.cx is None


    def test_modify_args_then_initialize(self):
        """Test you can modify arguments before initialization

        Also tests that further modifying the schema won't affect
        already initialized objects, but will affect future ones

        """
        class A:
            def __init__(self, x):
                self.x = x
        class B:
            def __init__(self, y):
                self.y = y
        sb = Schema(B, args=[1])
        sb2 = Schema(B, args=[10])
        sa = Schema(A, args=[sb])
        sa['x']['y'] += 1
        a = sa()
        sa['x']['y'] += 1
        assert sa['x']['y'] == 3
        a2 = sa()
        assert a2.x.y == 3
        assert a is not a2
        assert a.x.y == 2
        sa['x'] = sb2
        assert sa().x.y == 10

    def test_cache(self):
        """Test that cache is used; return already initialized objs"""
        class A:
            def __init__(self, x):
                self.x = x
        class B:
            def __init__(self, y):
                self.y = y
        sb = Schema(B, args=[1])
        sa = Schema(A, args=[sb])
        cache = {}
        a = sa(cache=cache)
        a2 = sa(cache=cache)
        assert a is a2
        assert a.x is a2.x
        a.x.y += 1
        assert a.x.y == a2.x.y


class TestSchemaArgs:

    def test_fails_duplicate_args(self):
        """Test specifying same argument twice will fail immediately"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        args = [1]
        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, args=args, kwargs=kwargs)

    def test_fails_missing_required_args(self):
        """Test missing argument will fail immediately"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)

    def test_fails_star_args(self):
        """Test star args are prohibited"""
        class A:
            def __init__(self, *args, y=3, **kwargs):
                if 'x' in kwargs:
                    self.x = kwargs['x']
                self.y = y

        kwargs = {'x': 1}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)


# Classes used for link testing below

class G:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class H:
    def __init__(self, z):
        self.z = z

class O:
    def __init__(self, val=-1):
        self.val = val
class T:
    def __init__(self, x):
        self.x = x
class A:
    def __init__(self, t1):
        self.t1 = t1
class B:
    def __init__(self, t2):
        self.t2 = t2
class C:
    def __init__(self, t3, t4):
        self.t3 = t3
        self.t4 = t4
class P:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class TestSchemaLinks:

    def test_initialize_link(self):
        """Test basic link usage

        Test uses the following structure:
        class G
            x: class H
              z: 1
            y: Link(x)
        """
        sh = Schema(H, args=[1])
        sg = Schema(G, args=[sh, Link(link_str='x')])
        g = sg.initialize()
        assert g.x is g.y

    def test_initialize_fails_non_existent_link(self):
        """Test link to non-existent referrent fails

        Test uses the following structure:
        class G
            x: Link(k)
            y: class H
              z: 2
        """
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':Link(link_str='k'), 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_fails_non_init_link(self):
        """Test link to non initialized referrent fails

        Test uses the following structure:
        class G
            x: Link(y)
            y: class H
              z: 2
        """
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':Link(link_str='y'), 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_fails_non_init_link_uncle(self):
        """Test link to non initialized referrent fails (diff nesting)

        This differs from previous because Link referrent is at a
        different level of nesting

        Test uses the following structure:
        class G
            x: class H
              z: Link(y)
            y: class H
              z: 2
        """
        sx = Schema(H, args=[Link(link_str='y')])
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':sx, 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_chained_link(self):
        """Test links chained

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val: <given>
            b: class B
              t2: Link(a[t1].x)
            c: class C
              t3: Link(a[t1].x)
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': 1})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': Link(link_str='a[t1].x')})
        sc = Schema(C, args=[Link(link_str='a[t1].x'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        # All links should be the same object in this case
        assert p.a.t1.x is p.b.t2
        assert p.c.t3 is p.c.t4
        assert p.c.t4 is p.a.t1.x

    def test_initialize_chained_link_to_attr(self):
        """Test link to attribute

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val: object()
            b: class B
              t2: Link(a[t1][x][val])
            c: class C
              t3: Link(a[t1].x)
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': object()})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': Link(link_str='a[t1][x][val]')})
        sc = Schema(C, args=[Link(link_str='a[t1].x'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        assert p.b.t2 is p.a.t1.x.val

    def test_initialize_link_thru_collection(self):
        """Test link to value through collections objects

        Tests both dict and list access via links

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val:
                    m: 9
                    p:
                      - 11
                      - 12
            b: class B
              t2: Link(a[t1][x][val][m])
            c: class C
              t3: Link(a[t1][x][val][p][1])
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': {'m': 9, 'p': [11, 12]}})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': Link(link_str='a[t1][x][val][m]')})
        sc = Schema(C, args=[Link(link_str='a[t1][x][val][p][1]'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        assert p.b.t2 == p.a.t1.x.val['m']
        assert p.c.t3 == p.a.t1.x.val['p'][1]

    def test_initialize_chained_copy_link(self):
        """Test copy links chained

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val: <given>
            b: class B
              t2: CopyLink(a[t1].x)
            c: class C
              t3: CopyLink(b[t2])
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': 1})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': CopyLink(link_str='a[t1].x')})
        sc = Schema(C, args=[CopyLink(link_str='b[t2]'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        # All links referrents should be different values because copy
        # links are used
        assert p.a.t1.x is not p.b.t2
        assert p.c.t3 is not p.c.t4
        assert p.c.t4 is not p.a.t1.x
        assert p.c.t3 is not p.b.t2
        # Except c.t4 and b.t2 should be the same because normal link
        # was used
        assert p.c.t4 is p.b.t2
        # Still the values should be the same
        assert p.a.t1.x.val == p.b.t2.val
        assert p.c.t3.val == p.c.t4.val
        assert p.c.t4.val == p.a.t1.x.val
        # An update should only affect the object changed
        p.b.t2.val = 2
        assert p.a.t1.x.val != p.b.t2.val
        assert p.c.t4.val == p.b.t2.val
        assert p.c.t3.val == p.a.t1.x.val
