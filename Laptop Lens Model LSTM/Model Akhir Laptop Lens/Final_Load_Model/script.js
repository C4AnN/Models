let lowChart, midChart, highChart;

async function loadModelFromFiles(jsonPath) {
    const model = await tf.loadLayersModel(jsonPath);
    return model;
}

async function loadModels() {
    const modelLow = await loadModelFromFiles('Model_All/best_model_tfjs_low/model.json');
    const modelMid = await loadModelFromFiles('Model_All/best_model_tfjs_mid/model.json');
    const modelHigh = await loadModelFromFiles('Model_All/best_model_tfjs_high/model.json');

    return {
        modelLow,
        modelMid,
        modelHigh
    };
}

function categorizePrice(price) {
    if (price <= 8000000) {
        return 'low';
    } else if (price > 8000000 && price <= 16000000) {
        return 'mid';
    } else if (price > 16000000) {
        return 'high';
    }
}

async function predictNextWeekSales(data, model, windowSize = 12) {
    const weeklySalesData = [];
    let currentWeek = [];
    for (const item of data) {
        if (currentWeek.length < 7) {
            currentWeek.push(item.sales);
        } else {
            weeklySalesData.push(currentWeek.reduce((a, b) => a + b, 0));
            currentWeek = [item.sales];
        }
    }
    if (currentWeek.length > 0) {
        weeklySalesData.push(currentWeek.reduce((a, b) => a + b, 0));
    }

    let inputData;
    if (weeklySalesData.length < windowSize) {
        inputData = new Array(windowSize - weeklySalesData.length).fill(0).concat(weeklySalesData);
    } else {
        inputData = weeklySalesData.slice(-windowSize);
    }

    let inputTensor = tf.tensor(inputData, [1, windowSize, 1]);

    const inputMin = inputTensor.min();
    const inputMax = inputTensor.max();
    let scaledInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin)).mul(2).sub(1);

    const dailyPredictions = [];
    for (let i = 0; i < 7; i++) {
        const predictedSalesScaled = model.predict(scaledInput);
        const predictedSalesActual = predictedSalesScaled.add(1).div(2).mul(inputMax.sub(inputMin)).add(inputMin).arraySync()[0][0];
        dailyPredictions.push(predictedSalesActual);

        inputData = inputData.slice(1).concat(predictedSalesActual);
        const newInputTensor = tf.tensor(inputData, [1, windowSize, 1]);
        scaledInput.dispose();
        inputTensor.dispose();
        inputTensor = newInputTensor;
        scaledInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin)).mul(2).sub(1);
    }

    return dailyPredictions;
}

function loadDataFromCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                const data = results.data.map(row => {
                    row.date = new Date(row.date);
                    if (isNaN(row.date)) {
                        console.error("Invalid date:", row.date);
                        return null;
                    }
                    row.category = categorizePrice(row.price);
                    return row;
                }).filter(row => row !== null);
                resolve(data);
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

function preparePlotDataLast6Months(data, predictedSales) {
    data.forEach(item => item.date = new Date(item.date));
    const weeklySales = {};
    data.forEach(item => {
        const week = new Date(item.date);
        week.setDate(week.getDate() - week.getDay());
        const weekKey = week.toISOString().slice(0, 10);
        if (!weeklySales[weekKey]) {
            weeklySales[weekKey] = 0;
        }
        weeklySales[weekKey] += item.sales;
    });

    const sortedWeeks = Object.keys(weeklySales).sort();
    const last6MonthsWeeks = sortedWeeks.slice(-24);
    const last6MonthsSales = last6MonthsWeeks.map(week => ({
        date: new Date(week),
        sales: weeklySales[week]
    }));

    // Sum of predicted sales for the next week
    const predictedSum = predictedSales.reduce((a, b) => parseInt(a) + parseInt(b), 0);

    const lastDate = last6MonthsSales[last6MonthsSales.length - 1].date;

    const nextWeekDate = new Date(lastDate);
    nextWeekDate.setDate(nextWeekDate.getDate() + 7);
    last6MonthsSales.push({
        date: nextWeekDate,
        sales: predictedSum
    });

    return last6MonthsSales;
}

function plotData(chartId, plotData, category, color, chartInstance) {
    const ctx = document.getElementById(chartId).getContext('2d');
    const labels = plotData.map(item => item.date.toISOString().slice(0, 10));
    const sales = plotData.map(item => item.sales);

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `Sales`,
                data: sales,
                borderColor: color,
                borderWidth: 2,
                fill: false,
                pointBackgroundColor: color,
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sales'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true
                },
                title: {
                    display: true,
                    text: `Sales Prediction for ${category} Category`
                },
                annotation: {
                    annotations: [{
                        type: 'line',
                        mode: 'vertical',
                        scaleID: 'x',
                        value: labels[labels.length - 9], // Adjusted for the additional predicted sum
                        borderColor: 'red',
                        borderWidth: 2,
                        label: {
                            content: 'Prediction Start',
                            enabled: true,
                            position: 'top'
                        }
                    }]
                },
                tooltip: {
                    callbacks: {
                        afterBody: function (context) {
                            if (context[0].dataIndex === labels.length - 1) {
                                return `Predicted total sales for the next week: ${sales[sales.length - 1]}`;
                            }
                        }
                    }
                }
            }
        }
    });

    return chartInstance;
}

async function main(file) {
    const { modelLow, modelMid, modelHigh } = await loadModels();

    const newData = await loadDataFromCSV(file);

    const newDataLow = newData.filter(item => item.category === 'low');
    const newDataMid = newData.filter(item => item.category === 'mid');
    const newDataHigh = newData.filter(item => item.category === 'high');

    const predictedSalesLow = await predictNextWeekSales(newDataLow, modelLow);
    const predictedSalesMid = await predictNextWeekSales(newDataMid, modelMid);
    const predictedSalesHigh = await predictNextWeekSales(newDataHigh, modelHigh);

    const plotDataLow = preparePlotDataLast6Months(newDataLow, predictedSalesLow);
    const plotDataMid = preparePlotDataLast6Months(newDataMid, predictedSalesMid);
    const plotDataHigh = preparePlotDataLast6Months(newDataHigh, predictedSalesHigh);

    lowChart = plotData('lowCategoryChart', plotDataLow, 'Low', 'blue', lowChart);
    midChart = plotData('midCategoryChart', plotDataMid, 'Mid', 'green', midChart);
    highChart = plotData('highCategoryChart', plotDataHigh, 'High', 'orange', highChart);

    const sumLow = predictedSalesLow.reduce((a, b) => a + b, 0);
    const sumMid = predictedSalesMid.reduce((a, b) => a + b, 0);
    const sumHigh = predictedSalesHigh.reduce((a, b) => a + b, 0);

}

document.getElementById('predictButton').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        main(fileInput.files[0]).catch(err => console.error(err));
    } else {
        alert('Please select a CSV file first.');
    }
});
