
let sidebarItems = document.querySelectorAll(".sidebar .item");
for (let item of sidebarItems) {
  item.id = "";
}
sidebarItems[8].id = "active";


function stepper(btn, elId){
    const myInput = document.getElementById(elId);

    let id = btn.getAttribute("id");
    let min = myInput.getAttribute("min");
    let max = myInput.getAttribute("max");
    let step = myInput.getAttribute("step");
    let val = myInput.value;

    
    let calcStep = (id == "increment-target-plan") ? (step*1) : (step * -1);
    let newValue = parseInt(val) + calcStep;

    if(newValue >= min && newValue <= max){

        myInput.value =  newValue;
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