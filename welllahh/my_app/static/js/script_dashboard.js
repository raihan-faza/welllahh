let ctx = document.getElementById("nutriChart");

let daily_nutri_type = document.querySelector("#graph-nutri-type").value;

let nutriChart;

function showNutriGraph(ctx) {
  const arrayRange = (start, stop, step) =>
    Array.from(
      { length: (stop - start) / step + 1 },
      (value, index) => start + index * step
    );

  let api_nutri_data = arrayRange(1, 32, 1);

  for (let i = 0; i < api_nutri_data.length; i++) {
    api_nutri_data[i] = 0;
  }

  const daily_nutrition = JSON.parse(
    document.getElementById("daily_nutrition").textContent
  );

  let groupedDailyNutrition = {
    CALORIE: [],
    CARBS: [],
    FAT: [],
    PROTEIN: [],
    DATE: [],
  };

  for (let i = 0; i < daily_nutrition.length; i++) {
    groupedDailyNutrition["CALORIE"].push(daily_nutrition[i].total_calories);
    groupedDailyNutrition["CARBS"].push(daily_nutrition[i].total_carbs);
    groupedDailyNutrition["FAT"].push(daily_nutrition[i].total_fat);
    groupedDailyNutrition["PROTEIN"].push(daily_nutrition[i].total_protein);
    groupedDailyNutrition["DATE"].push(daily_nutrition[i].date);
  }
  
  if (daily_nutri_type == "CALORIE") {
    for (let i = 0; i < groupedDailyNutrition["CALORIE"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];
      api_nutri_data[curr_date - 1] = groupedDailyNutrition["CALORIE"][i];
     
    }
  } else if (daily_nutri_type == "CARBS") {
    for (let i = 0; i < groupedDailyNutrition["CARBS"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];
      api_nutri_data[curr_date - 1] = groupedDailyNutrition["CARBS"][i];
    }
   
  } else if (daily_nutri_type == "FAT") {
    for (let i = 0; i < groupedDailyNutrition["FAT"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];
      api_nutri_data[curr_date - 1] = groupedDailyNutrition["FAT"][i];
    }
   
  } else if (daily_nutri_type == "PROTEIN") {
    for (let i = 0; i < groupedDailyNutrition["PROTEIN"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];
      api_nutri_data[curr_date - 1] = groupedDailyNutrition["PROTEIN"][i];
    }
   
  }

  // console.log("api_nutri_data: ", api_nutri_data);

  const labels = arrayRange(1, 32, 1);

  nutriChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: daily_nutri_type,
          data: api_nutri_data,
          fill: true,
          borderColor: "rgb(74,154,135)",

          tension: 0.1,
        },
      ],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}


showNutriGraph(ctx);


document
  .querySelector("#graph-nutri-type")
  .addEventListener("change", function () {
    daily_nutri_type = this.value;

    ctx = document.getElementById('nutriChart').getContext('2d');

    if (nutriChart) {
        nutriChart.destroy(); 
    }


    showNutriGraph(ctx);
});




const daily_nutrition_input = JSON.parse(
  document.getElementById("daily_nutrition_input").textContent
);



new gridjs.Grid({
  columns: ["Date", "Food", "Calorie", "Carbs", "Fat", "Protein", "Action"],
  data: daily_nutrition_input,
  style: {
    td: {
      border: "1px solid #ffe8e8",
    },
    th: {
      border: "1px solid #ffe8e8",
    },
  },
}).render(document.getElementById("wrapper"));


const bmi =  JSON.parse(
  document.getElementById("bmi").textContent
);


let bmiPercentage = ((bmi - 15) / (40-15) ) * 100

document.querySelector(".bmi-indicator").style.left = bmiPercentage + "%";

if (bmiPercentage > 60) {
  // obesitas
  document.querySelector(".health-status").style.backgroundColor = "#ff4242";
  document.querySelector(".health-status").textContent = "You aren't healthy. Obesity";
}


let nutri_health_status = document.querySelectorAll(".status-normal");
console.log("nutri_health_status: ", nutri_health_status);

for (let i=0; i < nutri_health_status.length; i++ ) {
  let curr_health_status = nutri_health_status[i].textContent;
  if (curr_health_status.toLowerCase() != "normal") {
    nutri_health_status[i].style.backgroundColor = "#ff4242";
    nutri_health_status[i].style.color = "#ffffff";
  }
}


const dateString = JSON.parse(
  document.getElementById("curr_date").textContent
);

const date = new Date(dateString);

const options = {
weekday: "long",
year: "numeric",
month: "long",
day: "numeric"
};

const formattedDate = date.toLocaleDateString("en-US", options);
let curentDate = document.getElementsByClassName("current-date");
curentDate[0].textContent = formattedDate;