import init, { create_mlp, train_mlp, predict_mlp } from './pkg/wasm_ia.js';

// Define dataset
let X = Array.from({ length: 500 }, () => [
    Math.random() * 2 - 1,
    Math.random() * 2 - 1
]);


// Creating Y array based on the condition
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

// Assigning points to the respective datasets based on Y values
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
window.update_page = (message) => {
    const [iteration, loss] = message.split(':').map(str => str.trim());
    lossChart.data.labels.push(iteration);
    lossChart.data.datasets[0].data.push(parseFloat(loss));
    lossChart.update();
};

// define global var model
let model;

async function run(learning_rate, nb_iter, layers) {
    await init();
    model = create_mlp(layers);

    let step = nb_iter / 30; // display only 30 loss

    await train_mlp(model, X, Y, learning_rate, nb_iter, step);

    // display prediction when training is over
    print_prediction();
}

//display prediction on chart
function print_prediction() {

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

    // Preparing data for Chart.js
    let data_predict = {
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

    // Assigning points to the respective datasets based on Y values
    for (let i = 0; i < X.length; i++) {
        if (Y[i][0] === 1) {
            data_predict.datasets[0].data.push({ x: X[i][0], y: X[i][1] });
        } else {
            data_predict.datasets[1].data.push({ x: X[i][0], y: X[i][1] });
        }
    }

    // Drawing the background
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

    // Creating the scatter plot
    const ctx_predict = document.getElementById('scatterPlot_predict').getContext('2d');
    const chart_predict = new Chart(ctx_predict, {
        type: 'scatter',
        data: data_predict,
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
    event.preventDefault();
    const learning_rate = parseFloat(document.getElementById('learning_rate').value);
    const nb_iter = parseInt(document.getElementById('nb_iter').value);
    const layers = document.getElementById('layers').value.split(',').map(Number);
    run(learning_rate, nb_iter, layers);
});