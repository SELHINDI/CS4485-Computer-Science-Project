// GDP Analysis & Predictor Tool - Enhanced Frontend with API Integration

// Global variables
let currentTheme = 'light';
let charts = {};
let currentCountry = 'USA';
let apiBaseUrl = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    try {
        showLoading(true);
        
        // Set up theme
        setupTheme();
        
        // Initialize navigation
        setupNavigation();
        
        // Set up event listeners
        setupEventListeners();
        
        // Load initial data
        await loadInitialData();
        
        // Update initial display
        await updateOverview();
        
        showLoading(false);
    } catch (error) {
        console.error('Error initializing app:', error);
        showError('Failed to initialize application');
        showLoading(false);
    }
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

function setupEventListeners() {
    // Country selection
    const countrySelect = document.getElementById('countrySelect');
    if (countrySelect) {
        countrySelect.addEventListener('change', async function() {
            currentCountry = this.value;
            await updateOverview();
        });
    }
    
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

async function loadInitialData() {
    try {
        // Load countries list
        const countriesResponse = await fetch('/api/countries');
        if (!countriesResponse.ok) {
            throw new Error('Failed to load countries');
        }
        const countries = await countriesResponse.json();
        
        // Populate country selectors
        populateCountrySelectors(countries);
        
        return countries;
    } catch (error) {
        console.error('Error loading initial data:', error);
        throw error;
    }
}

function populateCountrySelectors(countries) {
    const selectors = ['countrySelect', 'predictionCountry', 'country1', 'country2'];
    
    selectors.forEach(selectorId => {
        const selector = document.getElementById(selectorId);
        if (selector) {
            // Clear existing options
            selector.innerHTML = '';
            
            countries.forEach(country => {
                const option = document.createElement('option');
                option.value = country.code;
                option.textContent = country.name;
                selector.appendChild(option);
            });
        }
    });
}

async function updateOverview() {
    try {
        showLoading(true);
        
        // Load country data
        const countryData = await loadCountryData(currentCountry);
        
        // Update stats
        updateCountryStats(countryData);
        
        // Update overview chart
        await createOverviewChart(countryData);
        
        showLoading(false);
    } catch (error) {
        console.error('Error updating overview:', error);
        showError('Failed to load overview data');
        showLoading(false);
    }
}

async function loadCountryData(countryCode) {
    try {
        const response = await fetch(`/api/countries/${countryCode}`);
        if (!response.ok) {
            throw new Error(`Failed to load data for ${countryCode}: ${response.status}`);
        }
        const data = await response.json();
        console.log('Loaded country data:', data);
        return data;
    } catch (error) {
        console.error('Error loading country data:', error);
        throw error;
    }
}

function updateCountryStats(countryData) {
    try {
        // Update main stats
        document.getElementById('currentGDP').textContent = formatCurrency(countryData.current_gdp);
        document.getElementById('growthRate').textContent = countryData.growth_rate + '%';
        document.getElementById('perCapita').textContent = formatCurrency(countryData.per_capita);
        document.getElementById('worldRank').textContent = '#' + countryData.world_rank;
        
        // Update change indicators (simplified)
        document.getElementById('gdpChange').textContent = '+2.3%';
        document.getElementById('gdpChange').className = 'stat-change positive';
        
        document.getElementById('growthChange').textContent = '+0.5%';
        document.getElementById('growthChange').className = 'stat-change positive';
        
        document.getElementById('perCapitaChange').textContent = '+1.8%';
        document.getElementById('perCapitaChange').className = 'stat-change positive';
    } catch (error) {
        console.error('Error updating country stats:', error);
    }
}

async function createOverviewChart(countryData) {
    const ctx = document.getElementById('overviewChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (charts.overview) {
        charts.overview.destroy();
    }
    
    const historicalData = countryData.historical_data;
    if (!historicalData || historicalData.length === 0) {
        console.error('No historical data available');
        return;
    }
    
    // Ensure data is sorted by year
    const sortedData = historicalData.sort((a, b) => a.year - b.year);
    
    charts.overview = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sortedData.map(d => d.year.toString()),
            datasets: [{
                label: 'GDP (Trillion USD)',
                data: sortedData.map(d => parseFloat(d.gdp)),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    cornerRadius: 6
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

async function updateAnalysis() {
    try {
        showLoading(true);
        
        // Load country data
        const countryData = await loadCountryData(currentCountry);
        
        // Create analysis charts
        await createGrowthChart(countryData);
        await createSectorChart(countryData);
        
        // Update quarterly table
        updateQuarterlyTable(countryData);
        
        showLoading(false);
    } catch (error) {
        console.error('Error updating analysis:', error);
        showError('Failed to load analysis data');
        showLoading(false);
    }
}

async function createGrowthChart(countryData) {
    const ctx = document.getElementById('growthChart');
    if (!ctx) return;
    
    if (charts.growth) {
        charts.growth.destroy();
    }
    
    const historicalData = countryData.historical_data;
    if (!historicalData || historicalData.length === 0) {
        console.error('No historical data available for growth chart');
        return;
    }
    
    // Ensure data is sorted by year
    const sortedData = historicalData.sort((a, b) => a.year - b.year);
    
    charts.growth = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(d => d.year.toString()),
            datasets: [{
                label: 'Growth Rate (%)',
                data: sortedData.map(d => parseFloat(d.growth_rate)),
                backgroundColor: sortedData.map(d => 
                    parseFloat(d.growth_rate) >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                ),
                borderColor: sortedData.map(d => 
                    parseFloat(d.growth_rate) >= 0 ? '#10b981' : '#ef4444'
                ),
                borderWidth: 1,
                borderRadius: 3
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
                        color: 'rgba(0, 0, 0, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
    
    // Update insights
    updateGrowthInsights(sortedData);
}

function updateGrowthInsights(historicalData) {
    const insights = document.getElementById('growthInsights');
    if (!insights) return;
    
    const avgGrowth = historicalData.reduce((sum, d) => sum + d.growth_rate, 0) / historicalData.length;
    const recentGrowth = historicalData.slice(-3).reduce((sum, d) => sum + d.growth_rate, 0) / 3;
    const trend = recentGrowth > avgGrowth ? 'positive' : 'negative';
    
    insights.innerHTML = `
        <li>Average growth rate: ${avgGrowth.toFixed(1)}%</li>
        <li>Recent trend: ${trend} momentum</li>
        <li>Economic stability: ${avgGrowth > 2 ? 'Strong' : 'Moderate'}</li>
    `;
}

async function createSectorChart(countryData) {
    const ctx = document.getElementById('sectorChart');
    if (!ctx) return;
    
    if (charts.sector) {
        charts.sector.destroy();
    }
    
    const sectorData = countryData.sector_data;
    if (!sectorData || sectorData.length === 0) {
        console.error('No sector data available');
        return;
    }
    
    charts.sector = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: sectorData.map(d => d.sector),
            datasets: [{
                data: sectorData.map(d => parseFloat(d.percentage)),
                backgroundColor: [
                    '#3b82f6',
                    '#10b981',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ],
                borderWidth: 1,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
    
    // Update sector insights
    updateSectorInsights(sectorData);
}

function updateSectorInsights(sectorData) {
    const insights = document.getElementById('sectorInsights');
    if (!insights) return;
    
    const topSectors = sectorData.slice(0, 3);
    
    insights.innerHTML = topSectors.map(sector => 
        `<li>${sector.sector}: ${sector.percentage}%</li>`
    ).join('');
}

function updateQuarterlyTable(countryData) {
    const tbody = document.getElementById('quarterlyData');
    if (!tbody) return;
    
    const quarterlyData = countryData.quarterly_data;
    
    tbody.innerHTML = quarterlyData.map(quarter => `
        <tr>
            <td>${quarter.quarter}</td>
            <td>$${quarter.gdp}T</td>
            <td>${quarter.growth_rate}%</td>
            <td>${quarter.inflation}%</td>
            <td>${quarter.unemployment}%</td>
        </tr>
    `).join('');
}

async function updatePrediction() {
    // Initialize prediction section
    const resultsDiv = document.getElementById('predictionResults');
    if (resultsDiv) {
        resultsDiv.style.display = 'none';
    }
}

async function generatePrediction() {
    try {
        showLoading(true);
        
        const country = document.getElementById('predictionCountry').value;
        const years = parseInt(document.getElementById('predictionYears').value);
        const model = document.getElementById('predictionModel').value;
        
        // Make prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                country_code: country,
                years: years,
                model: model
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate prediction');
        }
        
        const predictionData = await response.json();
        
        // Create prediction chart
        await createPredictionChart(predictionData);
        
        // Update prediction stats
        updatePredictionStats(predictionData);
        
        // Show results
        const resultsDiv = document.getElementById('predictionResults');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
        
        showLoading(false);
    } catch (error) {
        console.error('Error generating prediction:', error);
        showError('Failed to generate prediction');
        showLoading(false);
    }
}

async function createPredictionChart(predictionData) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    
    if (charts.prediction) {
        charts.prediction.destroy();
    }
    
    const historicalData = predictionData.historical_data;
    const predictions = predictionData.predictions;
    
    const allData = [...historicalData, ...predictions];
    const labels = allData.map(d => d.year);
    const historicalValues = historicalData.map(d => d.gdp);
    const predictionValues = predictions.map(d => d.gdp);
    
    charts.prediction = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical GDP',
                    data: [...historicalValues, ...new Array(predictionValues.length).fill(null)],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                },
                {
                    label: 'AI Prediction',
                    data: [...new Array(historicalValues.length).fill(null), ...predictionValues],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#f59e0b',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
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

function updatePredictionStats(predictionData) {
    const lastPrediction = predictionData.predictions[predictionData.predictions.length - 1];
    
    document.getElementById('predictedGDP').textContent = formatCurrency(lastPrediction.gdp * 1000000000000);
    document.getElementById('modelAccuracy').textContent = (predictionData.r2_score * 100).toFixed(1) + '%';
    document.getElementById('confidenceLevel').textContent = predictionData.confidence_level.toFixed(0) + '%';
    document.getElementById('r2Score').textContent = predictionData.r2_score.toFixed(3);
}

async function updateComparison() {
    // Initialize comparison section
    const resultsDiv = document.getElementById('comparisonResults');
    if (resultsDiv) {
        resultsDiv.style.display = 'none';
    }
}

async function compareCountries() {
    try {
        showLoading(true);
        
        const country1 = document.getElementById('country1').value;
        const country2 = document.getElementById('country2').value;
        
        // Make comparison request
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                country1: country1,
                country2: country2
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to compare countries');
        }
        
        const comparisonData = await response.json();
        
        // Create comparison chart
        await createComparisonChart(comparisonData);
        
        // Update comparison table
        updateComparisonTable(comparisonData);
        
        // Show results
        const resultsDiv = document.getElementById('comparisonResults');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
        
        showLoading(false);
    } catch (error) {
        console.error('Error comparing countries:', error);
        showError('Failed to compare countries');
        showLoading(false);
    }
}

async function createComparisonChart(comparisonData) {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    
    if (charts.comparison) {
        charts.comparison.destroy();
    }
    
    const country1 = comparisonData.country1;
    const country2 = comparisonData.country2;
    
    const years = country1.historical_data.map(d => d.year);
    const country1Data = country1.historical_data.map(d => d.gdp);
    const country2Data = country2.historical_data.map(d => d.gdp);
    
    charts.comparison = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [
                {
                    label: country1.name,
                    data: country1Data,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                },
                {
                    label: country2.name,
                    data: country2Data,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
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

function updateComparisonTable(comparisonData) {
    const country1 = comparisonData.country1;
    const country2 = comparisonData.country2;
    
    // Update country names
    document.getElementById('country1Name').textContent = country1.name;
    document.getElementById('country2Name').textContent = country2.name;
    
    // Update GDP values
    document.getElementById('country1GDP').textContent = formatCurrency(country1.current_gdp);
    document.getElementById('country2GDP').textContent = formatCurrency(country2.current_gdp);
    
    // Calculate differences
    const gdpDiff = ((country1.current_gdp - country2.current_gdp) / country2.current_gdp * 100);
    document.getElementById('gdpDiff').textContent = (gdpDiff > 0 ? '+' : '') + gdpDiff.toFixed(1) + '%';
    
    // Update per capita values
    document.getElementById('country1PerCapita').textContent = formatCurrency(country1.per_capita);
    document.getElementById('country2PerCapita').textContent = formatCurrency(country2.per_capita);
    
    const perCapitaDiff = ((country1.per_capita - country2.per_capita) / country2.per_capita * 100);
    document.getElementById('perCapitaDiff').textContent = (perCapitaDiff > 0 ? '+' : '') + perCapitaDiff.toFixed(1) + '%';
    
    // Update growth rates
    document.getElementById('country1Growth').textContent = country1.growth_rate + '%';
    document.getElementById('country2Growth').textContent = country2.growth_rate + '%';
    
    const growthDiff = ((country1.growth_rate - country2.growth_rate) / country2.growth_rate * 100);
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

function showLoading(show) {
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = show ? 'flex' : 'none';
    }
}

function showError(message) {
    // Create a simple error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Export functions for potential external use
window.GDPAnalyzer = {
    formatCurrency,
    showLoading,
    showError,
    loadCountryData,
    generatePrediction,
    compareCountries
};