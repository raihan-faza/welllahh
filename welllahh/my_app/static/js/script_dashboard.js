let ctx = document.getElementById("nutriChart").getContext('2d');


let nutriChart;

function showNutriGraph(ctx, selectedMonth) {
  let daily_nutri_type = document.querySelector("#graph-nutri-type").value;

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

      let curr_data_date = groupedDailyNutrition["DATE"][i].split("-")
     
      
      if (curr_data_date[curr_data_date.length-2] == selectedMonth) {
        api_nutri_data[curr_date - 1] = groupedDailyNutrition["CALORIE"][i];
      }
    }
  } else if (daily_nutri_type == "CARBS") {
    for (let i = 0; i < groupedDailyNutrition["CARBS"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];

      let curr_data_date = groupedDailyNutrition["DATE"][i].split("-")

      if (curr_data_date[curr_data_date.length-2] == selectedMonth) {
        api_nutri_data[curr_date - 1] = groupedDailyNutrition["CARBS"][i];
      }
    
    }
   
  } else if (daily_nutri_type == "FAT") {
    for (let i = 0; i < groupedDailyNutrition["FAT"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];

      let curr_data_date = groupedDailyNutrition["DATE"][i].split("-")

      if (curr_data_date[curr_data_date.length-2] == selectedMonth) {
        api_nutri_data[curr_date - 1] = groupedDailyNutrition["FAT"][i];
      }
      
    }
   
  } else if (daily_nutri_type == "PROTEIN") {
    for (let i = 0; i < groupedDailyNutrition["PROTEIN"].length; i++) {
      let curr_date = groupedDailyNutrition["DATE"][i].split("-");
      curr_date = curr_date[curr_date.length - 1];

      let curr_data_date = groupedDailyNutrition["DATE"][i].split("-")

      if (curr_data_date[curr_data_date.length-2] == selectedMonth) {
      api_nutri_data[curr_date - 1] = groupedDailyNutrition["PROTEIN"][i];
      }
    }
   
  }


  const labels = arrayRange(1, 32, 1);

  if (nutriChart) nutriChart.destroy();

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




const nutriTypeSelect = document.getElementById("graph-nutri-type");
const dateSelect = document.getElementById("graph-date");

let selectedMonth = dateSelect.value;
function handleDropdownChange() {
   selectedMonth = dateSelect.value;

 

  ctx = document.getElementById('nutriChart').getContext('2d');

  if (nutriChart) {
      nutriChart.destroy(); 
  }


    showNutriGraph(ctx, selectedMonth);
}



nutriTypeSelect.addEventListener("change", handleDropdownChange);
dateSelect.addEventListener("change", handleDropdownChange);


let bmiLastWeek = JSON.parse(
  document.getElementById("bmi_last_week").textContent
)

let bmiLastYear = JSON.parse(
  document.getElementById("bmi_last_month").textContent
)

let bodyInfoLastWeek = JSON.parse(
  document.getElementById("body_last_week").textContent
)

let bodyInfoLastYear = JSON.parse(
  document.getElementById("body_last_month").textContent
)

let currBodyInfo = JSON.parse(
  document.getElementById("user_body_info_dict").textContent
)

let currBMI = JSON.parse(
  document.getElementById("curr_bmi").textContent
)

let bmiSelectDate = document.querySelector(".bmi-select");
bmiSelectDate.addEventListener("change", function() {
  let bmiDate = bmiSelectDate.value;
  let heightWeight = document.querySelectorAll(".measurement-value");
  let bmiValue = document.querySelector(".bmi-value");


  if (bmiDate == "last_week") {
    console.log("bmi change last_week")
    heightWeight[0].textContent = bodyInfoLastWeek.height;
    heightWeight[1].textContent = bodyInfoLastWeek.weight;
    bmiValue.textContent = bmiLastWeek;
  }else if  (bmiDate == "last_month") {
    console.log("bmi change last_month")
    heightWeight[0].textContent = bodyInfoLastYear.height;
    heightWeight[1].textContent = bodyInfoLastYear.weight;
    bmiValue.textContent = bmiLastYear
  }else {
    heightWeight[0].textContent = currBodyInfo.height;
    heightWeight[1].textContent = currBodyInfo.weight;
    bmiValue.textContent = currBMI;
  }


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

const bmiContainer =  document.querySelector(".bmi-value");


let bmiPercentage = ((bmi - 15) / (40-15) ) * 100

document.querySelector(".bmi-indicator").style.left = bmiPercentage + "%";

if (bmiPercentage > 60) {
  // obesitas
  document.querySelector(".health-status").style.backgroundColor = "#ff4242";
  document.querySelector(".health-status").textContent = "You aren't healthy. Obesity";
}

bmiContainer.addEventListener("change", function() {
  let newBmi = bmiContainer.value;
  let bmiPercentage = ((newBmi - 15) / (40-15) ) * 100
  document.querySelector(".bmi-indicator").style.left = bmiPercentage + "%";

  if (bmiPercentage > 60) {
    // obesitas
    document.querySelector(".health-status").style.backgroundColor = "#ff4242";
    document.querySelector(".health-status").textContent = "You aren't healthy. Obesity";
  }
});



let nutri_health_status = document.querySelectorAll(".status-normal");

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

const nutriTarget = JSON.parse(
  document.getElementById("nutri_target").textContent
);
const todayCalory=  JSON.parse(
  document.getElementById("today_calory").textContent
);

const todayProtein = JSON.parse(
  document.getElementById("today_protein").textContent
);

const todayCarbs = JSON.parse(
  document.getElementById("today_carbs").textContent
);

const todayFat = JSON.parse(
  document.getElementById("today_fat").textContent
);


// nutriTarget.calories
let nutriPieData = {
  
}




document.addEventListener("DOMContentLoaded",   function() {
  let calorieCtx = document.getElementById("caloriePie").getContext('2d');

  let carbCtx = document.getElementById("carbPie").getContext('2d');
  let proteinCtx = document.getElementById("proteinPie").getContext('2d');
  let fatCtx = document.getElementById("fatPie").getContext('2d'); 
  let caloriePie;
  let carbPie;
  let proteinPie;
  let fatPie;

  if (nutriChart) {
    nutriChart.destroy(); 
  }
  showNutriGraph(ctx, selectedMonth);
  
  if (caloriePie){
    caloriePie.destroy();
  }

  if (carbPie){
    carbPie.destroy();
  }

  if (proteinPie){
    proteinPie.destroy();
  }

  if (fatPie){
    fatPie.destroy();
  }

  caloriePie = new Chart(calorieCtx, {
    type: "doughnut",
    data: {
      labels: ["Calorie", "Remaining Calorie"],
      datasets: [
        {
          label: "Calorie",
          data: [todayCalory, nutriTarget.calories - todayCalory],
          backgroundColor: ["rgb(74,154,135)", "#f4473a"],
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: false,// harus false, kalau true nanti memory leak. gajelas librarynya
  },
  });



 carbPie = new Chart(carbCtx, {
  type: "doughnut",
  data: {
    labels: ["Carbs", "Remaining Carbs"],
    datasets: [
      {
        label: "Carbs",
        data: [todayCarbs, nutriTarget.carbs - todayCarbs],
        backgroundColor: ["rgb(74,154,135)", "#2196f3"],
        hoverOffset: 4,
      },
    ],
  },
  options: {
    responsive: false,// harus false, kalau true nanti memory leak. gajelas librarynya
},
});

 proteinPie = new Chart(proteinCtx, {
  type: "doughnut",
  data: {
    labels: ["Protein", "Remaining Protein"],
    datasets: [
      {
        label: "Protein",
        data: [todayProtein, nutriTarget.protein - todayProtein],
        backgroundColor: ["rgb(74,154,135)", "#f66c62"],
        hoverOffset: 4,
      },
    ],
  },
  options: {
    responsive: false,// harus false, kalau true nanti memory leak. gajelas librarynya
},
});

 fatPie = new Chart(fatCtx, {
  type: "doughnut",
  data: {
    labels: ["Fat", "Remaining Fat"],
    datasets: [
      {
        label: "Fat",
        data: [todayFat, nutriTarget.fat - todayFat],
        backgroundColor: ["rgb(74,154,135)", "#e79b38"],
        hoverOffset: 4,
      },
    ],
  },
  options: {
    responsive: false,// harus false, kalau true nanti memory leak. gajelas librarynya
},
});
});

