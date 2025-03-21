$(document).ready(function () {

    $("#div-textarea").show()
    $("#div-url").hide()
    $("#div-media").hide()

    // Show and Hide DIVs based on button click.
    $("#btn-prompt-text").click(() => {
        $("#div-textarea").show()
        $("#div-url").hide()
        $("#div-media").hide()
    });
    $("#btn-prompt-url").click(() => {
        $("#div-textarea").hide()
        $("#div-url").show()
        $("#div-media").hide()
    });
    $("#btn-prompt-media").click(() => {
        $("#div-textarea").hide()
        $("#div-url").hide()
        $("#div-media").show()
    });

    $("#btn-find-sentiment").click((e) => {
        e.preventDefault();

        let formData = new FormData();

        if ($("#div-textarea").is(":visible")) {
            formData.append("input", $("#form-input-textarea").val());
            formData.append("type", "text");
        } else if ($("#div-url").is(":visible")) {
            formData.append("input", $("#form-input-url").val());
            formData.append("type", "url");
        } else if ($("#div-media").is(":visible")) {
            let fileInput = $("#form-input-media")[0].files[0];
            let columnName = $("#csv-column").val().trim();

            if (!fileInput) {
                alert("Please upload a CSV file.");
                return;
            }
            if (!columnName) {
                alert("Please enter a column name.");
                return;
            }
            formData.append("input", fileInput);
            formData.append("column", columnName);
            formData.append("type", "media");
        }

        if (formData.get("input") !== '') {
            $("#sentiment-overlay").css("display", "block");

            $.ajax({
                type: "POST",
                url: '/',
                data: formData,
                contentType: false,
                processData: false,
            })
                .done((response, status, xhr) => {
                    let contentType = xhr.getResponseHeader("Content-Type");

                    if (contentType && contentType.includes("application/json")) {
                        // Handle JSON response (text & URL input)
                        let val_neg = Math.round(parseFloat(response["score_negative"]) * 10000) / 100;
                        let val_neu = Math.round(parseFloat(response["score_neutral"]) * 10000) / 100;
                        let val_pos = Math.round(parseFloat(response["score_positive"]) * 10000) / 100;

                        $("#sentiment-overlay-content").html(`
                        <h2 style="font-size: 2rem; font-weight: bold; text-align: center;">OVERALL SENTIMENT</h2>
                        <p id="prominent-sentiment" style="font-size: 1rem; text-align: center;">${response["prominent_sentiment"]}</p>
                        <div id="sentiment-info" style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center;">
                            <canvas id="sentiment-bar-chart" style="max-width: 700px; max-height: 500px; margin: 20px;"></canvas>
                            <canvas id="sentiment-pie-chart" style="max-width: 400px; max-height: 400px; margin: 20px;"></canvas>
                        </div>
                    `);

                        // Create Charts
                        createCharts(val_neg, val_neu, val_pos);

                    } else {
                        // Handle HTML response (CSV input)
                        $("body").html(response);
                    }
                })
                .fail((err) => {
                    console.log(err);
                });
        }
    });

    // Remove overlay
    $("#sentiment-overlay").click(() => {
        $("#sentiment-overlay").css("display", "none");
        $("#form-input-textarea").val("");
        $("#form-input-url").val("");
        $("#form-input-media").val("");
        window.location.reload();
    });

});

// Function to create sentiment charts
function createCharts(val_neg, val_neu, val_pos) {
    const barChart = new Chart($("#sentiment-bar-chart"), {
        type: 'bar',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{
                label: 'Sentiment Levels',
                data: [val_neg, val_neu, val_pos],
                backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(75, 192, 192, 0.8)'],
                borderColor: ['rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(75, 192, 192)'],
                borderWidth: 1,
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        font: {
                            size: 18,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 20,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 20,
                            weight: 'bold'
                        }
                    }
                }
            }
        }
    });

    const pieChart = new Chart($("#sentiment-pie-chart"), {
        type: 'pie',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{
                data: [val_neg, val_neu, val_pos],
                backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(75, 192, 192, 0.8)']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 16
                        }
                    }
                }
            }
        }
    });
}
