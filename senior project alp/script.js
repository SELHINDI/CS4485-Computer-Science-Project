// GDP Analysis & Predictor Tool - Main JavaScript File

// Global variables
let currentTheme = 'light';
let charts = {};
let gdpData = {};
let countryData = {};

// Sample GDP data for demonstration
const sampleGDPData = {
    USA: {
        name: 'United States',
        currentGDP: 25400000000000,
        growthRate: 3.2,
        perCapita: 76400,
        worldRank: 1,
        quarterlyData: [
            { quarter: 'Q1 2024', gdp: 25.1, growth: 2.8, inflation: 3.1, unemployment: 3.8 },
            { quarter: 'Q2 2024', gdp: 25.3, growth: 3.2, inflation: 2.9, unemployment: 3.7 },
            { quarter: 'Q3 2024', gdp: 25.4, growth: 3.5, inflation: 2.7, unemployment: 3.6 },
            { quarter: 'Q4 2024', gdp: 25.4, growth: 3.2, inflation: 2.8, unemployment: 3.5 }
        ],
        historicalData: [
            { year: 2015, gdp: 18.2, growth: 2.9 },
            { year: 2016, gdp: 18.7, growth: 1.7 },
            { year: 2017, gdp: 19.5, growth: 2.3 },
            { year: 2018, gdp: 20.5, growth: 2.9 },
            { year: 2019, gdp: 21.4, growth: 2.2 },
            { year: 2020, gdp: 20.9, growth: -2.2 },
            { year: 2021, gdp: 23.3, growth: 5.9 },
            { year: 2022, gdp: 25.5, growth: 2.1 },
            { year: 2023, gdp: 25.0, growth: 2.5 },
            { year: 2024, gdp: 25.4, growth: 3.2 }
        ],
        sectorData: [
            { sector: 'Services', percentage: 78.2, value: 19.8 },
            { sector: 'Manufacturing', percentage: 11.1, value: 2.8 },
            { sector: 'Agriculture', percentage: 0.9, value: 0.23 },
            { sector: 'Construction', percentage: 4.1, value: 1.04 },
            { sector: 'Other', percentage: 5.7, value: 1.45 }
        ]
    },
    CHN: {
        name: 'China',
        currentGDP: 17700000000000,
        growthRate: 5.1,
        perCapita: 12500,
        worldRank: 2,
        historicalData: [
            { year: 2015, gdp: 11.1, growth: 6.9 },
            { year: 2016, gdp: 11.2, growth: 6.7 },
            { year: 2017, gdp: 12.3, growth: 6.8 },
            { year: 2018, gdp: 13.9, growth: 6.7 },
            { year: 2019, gdp: 14.3, growth: 6.0 },
            { year: 2020, gdp: 14.7, growth: 2.2 },
            { year: 2021, gdp: 17.7, growth: 8.1 },
            { year: 2022, gdp: 17.9, growth: 3.0 },
            { year: 2023, gdp: 17.4, growth: 5.2 },
            { year: 2024, gdp: 17.7, growth: 5.1 }
        ]
    },
    JPN: {
        name: 'Japan',
        currentGDP: 4200000000000,
        growthRate: 1.0,
        perCapita: 33400,
        worldRank: 3,
        historicalData: [
            { year: 2015, gdp: 4.4, growth: 0.4 },
            { year: 2016, gdp: 4.9, growth: 0.5 },
            { year: 2017, gdp: 4.9, growth: 1.7 },
            { year: 2018, gdp: 5.0, growth: 0.3 },
            { year: 2019, gdp: 5.1, growth: 0.7 },
            { year: 2020, gdp: 4.9, growth: -4.3 },
            { year: 2021, gdp: 4.9, growth: 1.6 },
            { year: 2022, gdp: 4.2, growth: 1.0 },
            { year: 2023, gdp: 4.2, growth: 1.9 },
            { year: 2024, gdp: 4.2, growth: 1.0 }
        ]
    },
    DEU: {
        name: 'Germany',
        currentGDP: 4200000000000,
        growthRate: 0.2,
        perCapita: 50400,
        worldRank: 4,
        historicalData: [
            { year: 2015, gdp: 3.4, growth: 1.5 },
            { year: 2016, gdp: 3.5, growth: 2.2 },
            { year: 2017, gdp: 3.7, growth: 2.2 },
            { year: 2018, gdp: 3.9, growth: 1.1 },
            { year: 2019, gdp: 3.9, growth: 0.6 },
            { year: 2020, gdp: 3.8, growth: -4.6 },
            { year: 2021, gdp: 4.3, growth: 2.6 },
            { year: 2022, gdp: 4.3, growth: 1.8 },
            { year: 2023, gdp: 4.2, growth: -0.1 },
            { year: 2024, gdp: 4.2, growth: 0.2 }
        ]
    },
    IND: {
        name: 'India',
        currentGDP: 3700000000000,
        growthRate: 6.9,
        perCapita: 2600,
        worldRank: 5,
        historicalData: [
            { year: 2015, gdp: 2.1, growth: 7.4 },
            { year: 2016, gdp: 2.3, growth: 8.0 },
            { year: 2017, gdp: 2.7, growth: 6.8 },
            { year: 2018, gdp: 2.7, growth: 6.5 },
            { year: 2019, gdp: 2.9, growth: 4.0 },
            { year: 2020, gdp: 2.7, growth: -6.6 },
            { year: 2021, gdp: 3.2, growth: 8.7 },
            { year: 2022, gdp: 3.4, growth: 7.0 },
            { year: 2023, gdp: 3.5, growth: 7.2 },
            { year: 2024, gdp: 3.7, growth: 6.9 }
        ]
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set up theme
    setupTheme();
    
    // Initialize navigation
    setupNavigation();
    
    // Load sample data
    loadSampleData();
    
    // Initialize charts
    initializeCharts();
    
    // Set up event listeners
    setupEventListeners();
    
    // Update initial display
    updateOverview();
}

function setupTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    
    // Load saved theme or default to light
    currentTheme = localStorage.getItem('theme') || 'light';
    body.setAttribute('data-theme', currentTheme);
    
    // Update theme toggle icon
    updateThemeIcon();
    
    themeToggle.addEventListener('click', function() {
        currentTheme = currentTheme === 'light' ? 'dark' : 'light';
        body.setAttribute('data-theme', currentTheme);
        localStorage.setItem('theme', currentTheme);
        updateThemeIcon();
    });
}

function updateThemeIcon() {
    const icon = document.querySelector('#themeToggle i');
    icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');
    
    navButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetSection = this.getAttribute('data-section');
            
            // Update active nav button
            navButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Show target section
            sections.forEach(section => section.classList.remove('active'));
            document.getElementById(targetSection).classList.add('active');
            
            // Update content based on section
            switch(targetSection) {
                case 'overview':
                    updateOverview();
                    break;
                case 'analysis':
                    updateAnalysis();
                    break;
                case 'prediction':
                    updatePrediction();
                    break;
                case 'comparison':
                    updateComparison();
                    break;
            }
        });
    });
}

function loadSampleData() {
    gdpData = sampleGDPData.USA;
    countryData = sampleGDPData;
}

function initializeCharts() {
    // Overview chart
    const overviewCtx = document.getElementById('overviewChart');
    if (overviewCtx) {
        charts.overview = new Chart(overviewCtx, {
            type: 'line',
            data: {
                labels: gdpData.historicalData.map(d => d.year),
                datasets: [{
                    label: 'GDP (Trillion USD)',
                    data: gdpData.historicalData.map(d => d.gdp),
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }
            }
        });
    }
}

function setupEventListeners() {
    // Prediction controls
    const generatePredictionBtn = document.getElementById('generatePrediction');
    if (generatePredictionBtn) {
        generatePredictionBtn.addEventListener('click', generatePrediction);
    }
    
    // Comparison controls
    const compareCountriesBtn = document.getElementById('compareCountries');
    if (compareCountriesBtn) {
        compareCountriesBtn.addEventListener('click', compareCountries);
    }
}

function updateOverview() {
    // Update stats
    document.getElementById('currentGDP').textContent = formatCurrency(gdpData.currentGDP);
    document.getElementById('growthRate').textContent = gdpData.growthRate + '%';
    document.getElementById('perCapita').textContent = formatCurrency(gdpData.perCapita);
    document.getElementById('worldRank').textContent = '#' + gdpData.worldRank;
    
    // Update overview chart
    if (charts.overview) {
        charts.overview.data.labels = gdpData.historicalData.map(d => d.year);
        charts.overview.data.datasets[0].data = gdpData.historicalData.map(d => d.gdp);
        charts.overview.update();
    }
}

function updateAnalysis() {
    // Create growth chart
    createGrowthChart();
    
    // Create sector chart
    createSectorChart();
    
    // Update quarterly data table
    updateQuarterlyTable();
}

function createGrowthChart() {
    const ctx = document.getElementById('growthChart');
    if (!ctx) return;
    
    if (charts.growth) {
        charts.growth.destroy();
    }
    
    charts.growth = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: gdpData.historicalData.map(d => d.year),
            datasets: [{
                label: 'Growth Rate (%)',
                data: gdpData.historicalData.map(d => d.growth),
                backgroundColor: gdpData.historicalData.map(d => 
                    d.growth >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                ),
                borderColor: gdpData.historicalData.map(d => 
                    d.growth >= 0 ? '#10b981' : '#ef4444'
                ),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

function createSectorChart() {
    const ctx = document.getElementById('sectorChart');
    if (!ctx) return;
    
    if (charts.sector) {
        charts.sector.destroy();
    }
    
    charts.sector = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: gdpData.sectorData.map(d => d.sector),
            datasets: [{
                data: gdpData.sectorData.map(d => d.percentage),
                backgroundColor: [
                    '#2563eb',
                    '#10b981',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateQuarterlyTable() {
    const tbody = document.getElementById('quarterlyData');
    if (!tbody) return;
    
    tbody.innerHTML = gdpData.quarterlyData.map(quarter => `
        <tr>
            <td>${quarter.quarter}</td>
            <td>$${quarter.gdp}T</td>
            <td>${quarter.growth}%</td>
            <td>${quarter.inflation}%</td>
            <td>${quarter.unemployment}%</td>
        </tr>
    `).join('');
}

function updatePrediction() {
    // Initialize prediction with default values
    generatePrediction();
}

function generatePrediction() {
    const years = parseInt(document.getElementById('predictionYears').value);
    const model = document.getElementById('predictionModel').value;
    
    // Generate prediction data
    const predictionData = generatePredictionData(years, model);
    
    // Create prediction chart
    createPredictionChart(predictionData);
    
    // Update prediction stats
    updatePredictionStats(predictionData);
}

function generatePredictionData(years, model) {
    const historicalData = gdpData.historicalData;
    const lastYear = historicalData[historicalData.length - 1];
    const predictions = [];
    
    // Simple linear regression for prediction
    const x = historicalData.map((d, i) => i);
    const y = historicalData.map(d => d.gdp);
    
    // Calculate regression coefficients
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Generate predictions
    for (let i = 1; i <= years; i++) {
        const year = lastYear.year + i;
        let predictedGDP;
        
        switch(model) {
            case 'linear':
                predictedGDP = intercept + slope * (x.length + i - 1);
                break;
            case 'polynomial':
                predictedGDP = intercept + slope * (x.length + i - 1) + 0.1 * Math.pow(i, 2);
                break;
            case 'exponential':
                predictedGDP = lastYear.gdp * Math.pow(1.03, i); // 3% growth assumption
                break;
        }
        
        predictions.push({
            year: year,
            gdp: Math.max(0, predictedGDP),
            type: 'prediction'
        });
    }
    
    return {
        historical: historicalData,
        predictions: predictions,
        model: model,
        years: years
    };
}

function createPredictionChart(data) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    
    if (charts.prediction) {
        charts.prediction.destroy();
    }
    
    const allData = [...data.historical, ...data.predictions];
    const labels = allData.map(d => d.year);
    const historicalData = data.historical.map(d => d.gdp);
    const predictionData = data.predictions.map(d => d.gdp);
    
    charts.prediction = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical GDP',
                    data: [...historicalData, ...new Array(predictionData.length).fill(null)],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Predicted GDP',
                    data: [...new Array(historicalData.length).fill(null), ...predictionData],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

function updatePredictionStats(data) {
    const lastPrediction = data.predictions[data.predictions.length - 1];
    const currentGDP = data.historical[data.historical.length - 1].gdp;
    const averageGrowth = ((lastPrediction.gdp - currentGDP) / currentGDP * 100) / data.years;
    
    document.getElementById('predictedGDP').textContent = formatCurrency(lastPrediction.gdp * 1000000000000);
    document.getElementById('predictedGrowth').textContent = averageGrowth.toFixed(1) + '%';
    document.getElementById('confidenceLevel').textContent = '87%';
}

function updateComparison() {
    // Initialize comparison with default countries
    compareCountries();
}

function compareCountries() {
    const country1Code = document.getElementById('country1').value;
    const country2Code = document.getElementById('country2').value;
    
    const country1 = countryData[country1Code];
    const country2 = countryData[country2Code];
    
    if (!country1 || !country2) return;
    
    // Update comparison chart
    createComparisonChart(country1, country2);
    
    // Update comparison table
    updateComparisonTable(country1, country2);
}

function createComparisonChart(country1, country2) {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    
    if (charts.comparison) {
        charts.comparison.destroy();
    }
    
    const years = country1.historicalData.map(d => d.year);
    const country1Data = country1.historicalData.map(d => d.gdp);
    const country2Data = country2.historicalData.map(d => d.gdp);
    
    charts.comparison = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [
                {
                    label: country1.name,
                    data: country1Data,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4
                },
                {
                    label: country2.name,
                    data: country2Data,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

function updateComparisonTable(country1, country2) {
    // Update country names
    document.getElementById('country1Name').textContent = country1.name;
    document.getElementById('country2Name').textContent = country2.name;
    
    // Update GDP values
    document.getElementById('country1GDP').textContent = formatCurrency(country1.currentGDP);
    document.getElementById('country2GDP').textContent = formatCurrency(country2.currentGDP);
    
    // Calculate differences
    const gdpDiff = ((country1.currentGDP - country2.currentGDP) / country2.currentGDP * 100);
    document.getElementById('gdpDiff').textContent = (gdpDiff > 0 ? '+' : '') + gdpDiff.toFixed(1) + '%';
    
    // Update per capita values
    document.getElementById('country1PerCapita').textContent = formatCurrency(country1.perCapita);
    document.getElementById('country2PerCapita').textContent = formatCurrency(country2.perCapita);
    
    const perCapitaDiff = ((country1.perCapita - country2.perCapita) / country2.perCapita * 100);
    document.getElementById('perCapitaDiff').textContent = (perCapitaDiff > 0 ? '+' : '') + perCapitaDiff.toFixed(1) + '%';
    
    // Update growth rates
    document.getElementById('country1Growth').textContent = country1.growthRate + '%';
    document.getElementById('country2Growth').textContent = country2.growthRate + '%';
    
    const growthDiff = ((country1.growthRate - country2.growthRate) / country2.growthRate * 100);
    document.getElementById('growthDiff').textContent = (growthDiff > 0 ? '+' : '') + growthDiff.toFixed(1) + '%';
}

// Utility functions
function formatCurrency(value) {
    if (value >= 1e12) {
        return '$' + (value / 1e12).toFixed(1) + 'T';
    } else if (value >= 1e9) {
        return '$' + (value / 1e9).toFixed(1) + 'B';
    } else if (value >= 1e6) {
        return '$' + (value / 1e6).toFixed(1) + 'M';
    } else if (value >= 1e3) {
        return '$' + (value / 1e3).toFixed(1) + 'K';
    } else {
        return '$' + value.toFixed(0);
    }
}

function formatNumber(value) {
    return value.toLocaleString();
}

function calculateGrowthRate(current, previous) {
    return ((current - previous) / previous * 100).toFixed(1);
}

function calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    return (n * sumXY - sumX * sumY) / Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
}

// Export functions for potential external use
window.GDPAnalyzer = {
    formatCurrency,
    formatNumber,
    calculateGrowthRate,
    calculateCorrelation,
    generatePrediction: generatePredictionData
};