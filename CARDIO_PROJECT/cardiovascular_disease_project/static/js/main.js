// CardioML Interactive JavaScript & Theme Engine
document.addEventListener("DOMContentLoaded", function () {
    console.log("CardioML Engine Initialized.");

    // ==========================================
    // 1. LIGHT / DARK THEME TOGGLE LOGIC
    // ==========================================
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");

    // Load saved theme or default to dark
    const savedTheme = localStorage.getItem("cardio_theme") || "dark";
    setAppTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", function () {
            const currentTheme = document.documentElement.getAttribute("data-bs-theme") || "dark";
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            setAppTheme(newTheme);
            localStorage.setItem("cardio_theme", newTheme);
        });
    }

    function setAppTheme(theme) {
        document.documentElement.setAttribute("data-bs-theme", theme);
        if (themeIcon) {
            if (theme === "dark") {
                themeIcon.className = "fa-solid fa-sun text-warning";
                if (themeToggleBtn) themeToggleBtn.setAttribute("title", "Switch to Light Mode");
            } else {
                themeIcon.className = "fa-solid fa-moon text-primary";
                if (themeToggleBtn) themeToggleBtn.setAttribute("title", "Switch to Dark Mode");
            }
        }
    }

    // ==========================================
    // 2. LIVE BMI & BP CALCULATOR LISTENERS
    // ==========================================
    const heightInput = document.querySelector('input[name="height"]');
    const weightInput = document.querySelector('input[name="weight"]');
    const apHiInput = document.querySelector('input[name="ap_hi"]');
    const apLoInput = document.querySelector('input[name="ap_lo"]');
    const bmiPreview = document.getElementById("liveBmiPreview");
    const bpPreview = document.getElementById("liveBpPreview");

    function updateLiveBmi() {
        if (!heightInput || !weightInput || !bmiPreview) return;
        const h = parseFloat(heightInput.value);
        const w = parseFloat(weightInput.value);

        if (h > 50 && w > 20) {
            const bmi = (w / ((h / 100) ** 2)).toFixed(1);
            let category = "Normal";
            let colorClass = "text-success";

            if (bmi < 18.5) { category = "Underweight"; colorClass = "text-info"; }
            else if (bmi >= 25 && bmi < 30) { category = "Overweight"; colorClass = "text-warning"; }
            else if (bmi >= 30) { category = "Obese"; colorClass = "text-danger"; }

            bmiPreview.innerHTML = `Live BMI: <strong class="${colorClass}">${bmi} kg/m² (${category})</strong>`;
        }
    }

    function updateLiveBp() {
        if (!apHiInput || !apLoInput || !bpPreview) return;
        const hi = parseInt(apHiInput.value);
        const lo = parseInt(apLoInput.value);

        if (hi > 40 && lo > 30) {
            let bpStatus = "Normal";
            let colorClass = "text-success";

            if (hi < 120 && lo < 80) {
                bpStatus = "Normal (<120/80)";
                colorClass = "text-success";
            } else if (hi >= 120 && hi <= 129 && lo < 80) {
                bpStatus = "Elevated (120-129/<80)";
                colorClass = "text-info";
            } else if ((hi >= 130 && hi <= 139) || (lo >= 80 && lo <= 89)) {
                bpStatus = "Stage 1 Hypertension";
                colorClass = "text-warning";
            } else {
                bpStatus = "Stage 2 Hypertension";
                colorClass = "text-danger";
            }

            bpPreview.innerHTML = `Live BP Category: <strong class="${colorClass}">${bpStatus}</strong>`;
        }
    }

    if (heightInput) heightInput.addEventListener("input", updateLiveBmi);
    if (weightInput) weightInput.addEventListener("input", updateLiveBmi);
    if (apHiInput) apHiInput.addEventListener("input", updateLiveBp);
    if (apLoInput) apLoInput.addEventListener("input", updateLiveBp);

    updateLiveBmi();
    updateLiveBp();
});
