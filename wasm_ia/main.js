const { default: init, predict_mlp } = await import('./pkg/wasm_ia.js');
await init()

// Define global variables
let model;
let chart_predict;

// Define dataset
let X = Array.from({ length: 500 }, () => [
    Math.random() * 2 - 1,
    Math.random() * 2 - 1
]);

// Define labels
let Y = X.map(p => [(Math.abs(p[0]) <= 0.3 || Math.abs(p[1]) <= 0.3) ? 1 : -1]);

// Create datasets for chart
let data = {
    datasets: [
        {
            label: 'Blue Class',
            data: [],
            backgroundColor: 'blue'
        },
        {
            label: 'Red Class',
            data: [],
            backgroundColor: 'red'
        }
    ]
};

// Assigning points to the chart datasets based on Y values
for (let i = 0; i < X.length; i++) {
    if (Y[i][0] === 1) {
        data.datasets[0].data.push({ x: X[i][0], y: X[i][1] });
    } else {
        data.datasets[1].data.push({ x: X[i][0], y: X[i][1] });
    }
}

// Creating a chart with the dataset
const datasetChart = document.getElementById('scatterPlot').getContext('2d');
new Chart(datasetChart, {
    type: 'scatter',
    data: data,
    options: {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom'
            },
            y: {
                type: 'linear'
            }
        }
    }
});

// Creating a chart for the loss
const ctx = document.getElementById('lossChart').getContext('2d');
const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Loss',
            data: [],
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    },
    options: {
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Iteration'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Loss'
                }
            }
        }
    }
});

// Function called by the Rust to update the loss chart
window.update_loss_chart = (message) => {
    const [iteration, loss] = message.split(':').map(str => str.trim());
    lossChart.data.labels.push(iteration);
    lossChart.data.datasets[0].data.push(parseFloat(loss));
    lossChart.update();
};


// Function that call wasm to create & train model
function run(learning_rate, nb_iter, layers) {
    document.getElementById("prediction_div").hidden = true;

    let step = nb_iter / 30; // display only 30 loss

    // Create a worker to run training in parallel
    const worker = new Worker('./worker.js');
    worker.onmessage = function (e) {
        const { type, data } = e.data;
        if (type === 'progress') {
            console.log('Received message from worker:', data);
            update_loss_chart(data)
        }
        else if (type === 'updatedModel') {
            console.log('Received updated model from worker');
            model = data;
            // display prediction when training is over
            print_prediction();
        }
    };

    // Send message to the worker to create and train the model
    worker.postMessage({
        layers,
        X,
        Y,
        learning_rate,
        nb_iter,
        step
    });    
}


// Display prediction on chart
async function print_prediction() {
    document.getElementById("prediction_div").hidden = false;

    // define a grid of point to predict
    const step = 0.02;
    let gridPoints = [];
    for (let x = -1; x <= 1; x += step) {
        for (let y = -1; y <= 1; y += step) {
            gridPoints.push([x, y]);
        }
    }

    // Predicting the class for each point in the grid
    let backgroundColors = gridPoints.map(p => predict_mlp(model, [p[0], p[1]], 1)[0] > 0 ? '#5787f6' : '#fcaf9e');

    // Drawing the background that will be the prediction area
    const backgroundPlugin = {
        id: 'backgroundPlugin',
        beforeDraw: (chart) => {
            const ctx = chart.ctx;
            const chartArea = chart.chartArea;

            for (let i = 0; i < gridPoints.length; i++) {
                const [x, y] = gridPoints[i];
                const color = backgroundColors[i];

                const pixelX = chart.scales.x.getPixelForValue(x);
                const pixelY = chart.scales.y.getPixelForValue(y);

                ctx.fillStyle = color;
                ctx.fillRect(pixelX - step * chart.width / 2, pixelY - step * chart.height / 2, step * chart.width, step * chart.height);
            }
        }
    };

    // Creating the scatter plot for the prediction
    const ctx_predict = document.getElementById('scatterPlot_predict').getContext('2d');    
    if (chart_predict) {
        chart_predict.destroy();
    }
    chart_predict = new Chart(ctx_predict, {
        type: 'scatter',
        data: data,
        options: {
            plugins: {
                filler: {
                    propagate: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    type: 'linear'
                }
            },
            animation: {
                duration: 0
            }
        },
        plugins: [backgroundPlugin]
    });
}


// Event listener to run training when form is submited
document.getElementById('trainingForm').addEventListener('submit', (event) => {
    lossChart.data.labels = []
    lossChart.data.datasets[0].data = []
    event.preventDefault();
    const learning_rate = parseFloat(document.getElementById('learning_rate').value);
    const nb_iter = parseInt(document.getElementById('nb_iter').value);
    const layers = document.getElementById('layers').value.split(',').map(Number);
    run(learning_rate, nb_iter, layers);
});


// Event listener to run prediction when form is submited
document.getElementById('prediction_form').addEventListener('submit', (event) => {
    event.preventDefault();
    const x = parseFloat(document.getElementById('x').value);
    const y = parseFloat(document.getElementById('y').value);
    const result = predict_mlp(model, [x, y])[0]
    const percent = (((result + 1)/2) * 100).toFixed(2)
    const prediction_result = document.getElementById('prediction_result')
    if (result < 0) {
        prediction_result.style.color = 'red'
        prediction_result.innerText = `Red class at ${percent}%`
    }
    else {
        prediction_result.style.color = 'blue'
        prediction_result.innerText = `Blue class at ${percent}%`
    }
});