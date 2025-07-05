document.addEventListener('DOMContentLoaded', () => {
    // Traffic patterns chart for analysis page
    if (document.getElementById('trafficPatternsChart')) {
        renderTrafficPatternsChart();
    }

    // Form validation for prediction
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', validatePredictionForm);
    }

    // Initialize any tooltips
    initTooltips();
});

// Render interactive traffic patterns chart
function renderTrafficPatternsChart() {
    const ctx = document.getElementById('trafficPatternsChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 24}, (_, i) => i), // Hours 0-23
            datasets: [{
                label: 'Average Traffic Volume',
                data: [50, 40, 35, 30, 25, 30, 45, 80, 120, 150, 160, 165, 
                       160, 155, 150, 155, 170, 180, 160, 130, 100, 80, 65, 55],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Typical Daily Traffic Volume Pattern',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw} vehicles`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Traffic Volume (vehicles)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Validate prediction form inputs
function validatePredictionForm(e) {
    let isValid = true;
    const errorMessages = [];
    
    // Validate temperature
    const temp = document.getElementById('temperature').value;
    if (temp < -20 || temp > 50) {
        errorMessages.push('Temperature must be between -20°C and 50°C');
        isValid = false;
    }
    
    // Validate rain
    const rain = document.getElementById('rain').value;
    if (rain < 0 || rain > 100) {
        errorMessages.push('Rain must be between 0mm and 100mm');
        isValid = false;
    }
    
    // Validate hour
    const hour = document.getElementById('hour').value;
    if (hour < 0 || hour > 23) {
        errorMessages.push('Hour must be between 0 and 23');
        isValid = false;
    }
    
    // Validate cloud cover
    const clouds = document.getElementById('cloud_cover').value;
    if (clouds < 0 || clouds > 100) {
        errorMessages.push('Cloud cover must be between 0% and 100%');
        isValid = false;
    }
    
    if (!isValid) {
        e.preventDefault();
        alert('Please fix the following errors:\n\n' + errorMessages.join('\n'));
    }
}

// Initialize tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Update prediction form based on time inputs
function updateTimeDependentFields() {
    const hour = parseInt(document.getElementById('hour').value);
    const weekday = parseInt(document.getElementById('weekday').value);
    
    if (!isNaN(hour)) {
        // Auto-set is_daytime
        document.getElementById('daytime_yes').checked = (hour >= 6 && hour <= 18);
        document.getElementById('daytime_no').checked = !(hour >= 6 && hour <= 18);
        
        // Auto-set is_peak_hour
        const isPeak = (hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 18);
        document.getElementById('peak_yes').checked = isPeak;
        document.getElementById('peak_no').checked = !isPeak;
    }
    
    if (!isNaN(weekday)) {
        // Auto-set is_weekend
        document.getElementById('weekend_yes').checked = (weekday >= 5);
        document.getElementById('weekend_no').checked = (weekday < 5);
    }
}

// Add event listeners for auto-updating fields
document.addEventListener('DOMContentLoaded', () => {
    const hourInput = document.getElementById('hour');
    const weekdayInput = document.getElementById('weekday');
    
    if (hourInput) hourInput.addEventListener('change', updateTimeDependentFields);
    if (weekdayInput) weekdayInput.addEventListener('change', updateTimeDependentFields);
});