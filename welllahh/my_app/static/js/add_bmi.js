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