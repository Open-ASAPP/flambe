var nodesArray, nodes, edges, network;

$(function () {
    $('[data-toggle="tooltip"]').tooltip()
})

function updateProgressBar(fraction) {
    $("#progress").css("width", 100 * fraction + "%");
}

function updateCountTable(state) {
    var values = $.map(state.block_state, function(value, key) { return value  });
    var pending = values.filter(function(i) { return i === 0;  });
    var running = values.filter(function(i) { return i === 1;  });
    var success = values.filter(function(i) { return i === 2;  });
    var failure = values.filter(function(i) { return i === 3;  });
    console.log(failure)
    $("#pending-count").text(pending.length + running.length);
    $("#success-count").text(success.length);
    $("#failure-count").text(failure.length);
    $("#total-count").text(state.total);
}

function updateTopAlerts(state) {
    if (state.finished) {
        var values = $.map(state.block_state, function(value, key) { return value  });
        var allTrue = values.every(function(i) { return i === 2;  });
        if (allTrue){
            if (state.factories_num == 0) {
                $("#top-alert").attr("class", "alert alert-success").html("<strong>Wo Hoo!</strong> Experiment <code>" + state.name +"</code> ended successfully");
            } else {
                var downloadDiv = 
                    '<div>'+
                        '<a class="btn btn-success" style="margin: 1em" href="download" target="_blank" role="button">Download</a>'+
                        '<a class="btn btn-success" style="margin: 1em" href="download_logs" target="_blank" role="button">Download Logs</a>'+
                    '</div>'
                $("#top-alert").attr("class", "alert alert-success").html("<strong>Wo Hoo!</strong> Experiment <code>" + state.name + "</code> ended successfully" + downloadDiv);
            }
        } else {
            $("#top-alert").attr("class", "alert alert-danger").html("<strong>Oh snap!</strong> Some of the stages in Experiment <code>" + state.name +"</code> ended up with errors. Check the logs for more information");
        }
    } else {
        console.log(state.finished);
        $("#top-alert").attr("class", "alert alert-info").html("Experiment <code>" + state.name + "</code> in progress");
    }
}

function updateEnv(state) {
    if (state.factories_num == 0) {
        $("#env").attr("class", "badge badge-secondary").text("Local Experiment");
    } else {
        $("#env").attr("class", "badge badge-secondary").text("Running remotely in " + state.factories_num + " factories");
    }
}

function updateVis(state) {
    Object.keys(state.dependency_dag).forEach(function(key) {
        if (key in state.block_state){
            if (state.block_state[key] === 0) {
                var color = "white";
                var extra = "badge-secondary"
                var text = "Pending"
            } else if (state.block_state[key] === 1) {
                var color = " #4da3ff";
                var extra = "badge-info"
                var text = "Running"
            } else if (state.block_state[key] === 2) {
                var color = " #5bd778";
                var extra = "badge-success"
                var text = "Completed"

                $("#" + key).find("#time").html("<h4>Execution time: <b>"+parseFloat(state.time_lapses[key]).toFixed(2) + "</b> secs</h4>");
            } else if (state.block_state[key] === 3) {
                var color = "#e46774";
                var extra = "badge-danger"
                var text = "Failed"
            }
            nodes.update([{id:key, color:{background:color, highlight: {background: color}}}]);
            $("#" + key).find("#badge-status").attr('class', 'badge ' + extra).text(text)
        }

        if (tb_path != null) {
            var href = tb_path + "/#scalars&regexInput=" + key
            $("#" + key).find("#tb-button-div").html('<a class="btn btn-outline-dark" id="tb-button" style="margin: 1em" href="' + href + '" target="_blank">Tensorboard</a>');
        }
        if (state.factories_num != 0 && state.block_state[key] > 1) {
            var href = "download/" + key;
            $("#" + key).find("#download-button-div").html('<a class="btn btn-outline-dark" id="tb-button" style="margin: 1em" href="' + href + '" target="_blank">Download</a>');

            var href = "download_logs/" + key;
            $("#" + key).find("#download-logs-button-div").html('<a class="btn btn-outline-dark" id="tb-button" style="margin: 1em" href="' + href + '" target="_blank">Download Logs</a>');
        }
        if (key in state.variants){
            $("#" + key).find("#hparams").html("")
            state.variants[key].forEach(function(v) {
                var div = "<div class='col-sm-5 mx-auto' style='margin: 1em;'><div class='card border-dark'><div class='card-body'><table class='mx-auto table'><tbody>"
                var tb_href = tb_path + "/#scalars&regexInput=" + key + "%2F";
                var download_param = "";
                Object.entries(v).forEach(function(entry) {
                    div += "<tr><td><b>" + entry[0] + "</b></td><td>" + entry[1] + "</td></tr>";
                    tb_href += entry[0] + "%3D" + entry[1] + "%2C";
                    download_param += entry[0] + "=" + entry[1] + ",";
                });
                tb_href = tb_href.substring(0, tb_href.length - 3)  // Remove last ',' ('%2C')
                download_param = download_param.substring(0, download_param.length - 1)  // Remove last ','
                var href = tb_path + "/#scalars&regexInput=" + key
                if (state.factories_num != 0 && state.block_state[key] > 1) {
                    div += "</tbody></table>"+
                        "<div class='row' align='center'>"+
                            "<a class='btn btn-sm btn-outline-secondary' id='tb-button' style='margin: 0.5em' href=download/" + key + "/" + download_param + " target='_blank'>Download</a>"+
                            "<a class='btn btn-sm btn-outline-secondary' id='tb-button' style='margin: 0.5em' href=download_logs/" + key + "/" + download_param + " target='_blank'>Download Logs</a>"+
                            "<a class='btn btn-sm btn-outline-secondary' id='tb-button' style='margin: 0.5em' href=" + tb_href + " target='_blank'>Tensorboard</a>"+
                        "</div>"+
                        "</div></div>";
                } else {
                    div += "</tbody></table>"+
                        "<div class='row' align='center'>"+
                            "<a class='btn btn-sm btn-outline-secondary' id='tb-button' style='margin: 0.5em' href=" + tb_href + " target='_blank'>Tensorboard</a>"+
                        "</div>"+
                        "</div></div>";
                }
                $("#" + key).find("#hparams").append(div);
            });
            if (state.variants[key].length > 0) {
                $("#" + key).find("#variants-title").text("Variants:");
                $("#" + key).find("#variants-num").html("<h4># of variants: " + state.variants[key].length + "</h4>");
            } else {
                $("#" + key).find("#variants-num").html("<h4>Block has no variants</h4>");
            }
        }

    });
}

function createModals(state) {
    Object.keys(state.dependency_dag).forEach(function(key) {
        var div = 
            '<div class="modal fade" id="' + key +'" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">' +
                '<div class="modal-dialog" role="document">'+
                    '<div class="modal-content">'+
                        '<div class="modal-header">'+
                            '<h3 class="modal-title" id="exampleModalLabel">' + key + '</h3>'+
                            '<button type="button" class="close" data-dismiss="modal" aria-label="Close">'+
                                '<span aria-hidden="true">&times;</span>'+
                            '</button>'+
                        '</div>'+
                        '<div class="modal-body">'+
                            '<div class="container">'+
                                '<div class="row">'+
                                    '<div class="col-md-6">'+
                                        '<h3>State: <span id="badge-status" class="badge badge-info" style="margin: 1em">Pending</span></h3>'+
                                    '</div>'+
                                    '<div class="row col-md-6" align="left">'+
                                        '<div class="" id="download-logs-button-div">'+
                                        '</div>'+
                                        '<div class="" id="download-button-div">'+
                                        '</div>'+
                                        '<div class="" id="tb-button-div">'+
                                        '</div>'+
                                    '</div>'+
                                '</div>'+
                                '<p class="card-text" id="duration"></p>'+
                            '</div>'+
                            '<div class="container" id="variants-num">'+
                            '</div>'+
                            '<div class="container" id="time">'+
                            '</div>'+
                            '<div class="container">'+
                                '<h4 class="modal-title" id="variants-title"></h4>'+
                                '<div class="row" id="hparams">'+
                            '</div>'+
                        '</div>'+
                    '</div>'+
                '</div>'+
            '</div>'
        var $newdiv = $( div );
        $("body").append($newdiv);
    });
}

function startVis(state) {
    nodesArray = []
    edges = [];
    Object.keys(state.dependency_dag).forEach(function(key) {
        nodesArray.push(
            {
                id: key,
                shape: 'box', label: key,
                color: {
                    border: 'black',
                    highlight: {
                        border: 'black',
                        background: 'white'
                    },
                },
                scaling: {
                    label: {
                        enabled: true,
                        min: 10,
                        max: 10
                    }
                }
            }
        );

    });
    Object.keys(state.dependency_dag).forEach(function(key) {
        state.dependency_dag[key].forEach(function(t) {
            edges.push(
                {
                    from: t,
                    to: key,
                    arrows:'to',
                    color: {
                        color:'#848484',
                        highlight:'#848484',
                        inherit: 'from',
                        opacity:1.0
                    },
                }
            );
        });
    });

    nodes = new vis.DataSet(nodesArray);
    var container = document.getElementById('mynetwork');

    $("#mynetwork").css({"width": "100%", "height": "30em", "border": "0.1em solid gray", "margin-bottom": "5em", "margin-top": "3em"});
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {

        nodes: {
            size:100,
            color: {
                background: 'white'
            },
            font:{color:'black', "size": 25},
        },
        interaction: {
            dragNodes: false,
        },
        layout:{randomSeed:2}

    };
    network = new vis.Network(container, data, options);

    network.on("click", function (params) {
        console.log(params['nodes'])
        if (params['nodes'].length > 0) {
            $('#' +params['nodes'][0]).modal('toggle');
        }
        params.event = "[original event]";

    });
    network.on("hoverNode", function (params) {
        params.event = "[original event]";
    });
}

function update() {
    $.ajax({url: "state", success: function(state){
        state = JSON.parse(state)
        updateProgressBar(state.done / state.total);
        updateTopAlerts(state);
        updateEnv(state);
        updateCountTable(state);
        if (network === undefined && state !== null){
            startVis(state);
            createModals(state);
        }
        updateVis(state);
    }, error: function(){
        $("#top-alert").attr("class", "alert alert-danger").html("<strong>Oops!</strong> Results not available");
    }});
}
