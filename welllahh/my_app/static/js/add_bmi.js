function updateValue(field, change) {
    const element = document.getElementById(field);
    let value = parseInt(element.value);

    if (field === "weight") {
      value = Math.max(30, Math.min(200, value + change));
    } else if (field === "age") {
      value = Math.max(1, Math.min(120, value + change));
    }

    element.value = value;
  }

  function updateHeight(value, type) {
    document.getElementById(type).value = value;
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