// import bot from './bot.svg';
// import user from './user.svg';
const bot = "{% static 'img/bot.svg' %}"
const user = "{% static 'img/user.svg' %}"

const submitButton = document.querySelector('#submit');
const outputElement = document.querySelector('#output');
const inputElement = document.querySelector('textarea');
const historyElement = document.querySelector('.history');
const buttonNewChatElement = document.querySelector('button');

const chatContainer = document.querySelector('.chat-container')


function changeInput(value) {
    const inputElement = document.querySelector('textarea');
    inputElement.value = value;
}


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}



let loadInterval

function loader(element) {
    element.textContent = ''

    loadInterval = setInterval(() => {
        element.textContent += '.';

        if (element.textContent === '....') {
            element.textContent = '';
        }
    }, 300);
}

function typeText(element, text) {
    let index = 0

    let interval = setInterval(() => {
        if (index < text.length) {
            element.innerHTML += text.charAt(index)
            index++
        } else {
            clearInterval(interval)
        }
    }, 20)
}


function generateUniqueId() {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);

    return `id-${timestamp}-${hexadecimalString}`;
}


function chatStripe(isAi, value, uniqueId) {
    return (
        `
        <div class="wrapper ${isAi && 'ai'}">
            <div class="chat">
                <div class="profile">
                    <img 
                      src=${isAi ? bot : user} 
                      alt="${isAi ? 'bot' : 'user'}" 
                    />
                </div>
                <div class="message" id=${uniqueId}>${value}</div>
            </div>
        </div>
    `
    )
}




const csrftoken = getCookie('csrftoken');

async function getMessage() {

    const uniqueId = generateUniqueId()
    chatContainer.innerHTML += chatStripe(true, " ", uniqueId)

    chatContainer.scrollTop = chatContainer.scrollHeight;

    const messageDiv = document.getElementById(uniqueId)

    loader(messageDiv)


    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken
        },
        body: JSON.stringify({ 
            question: inputElement.value,
        })
    }

    try{

        const response =  await fetch("http://localhost:8000/my_app/ai/chatbot", options)

      
        clearInterval(loadInterval)
        messageDiv.innerHTML = " "

        let data;
        if (response.ok) {
            data = await response.json();
            const parsedData = data.chatbot_message.trim()

            typeText(messageDiv, parsedData)
        } else {
            const err = await response.text()

            messageDiv.innerHTML = "Something went wrong"
            alert(err)
        }


        
        outputElement.textContent = data.chatbot_message
        if (data.chatbot_message) {
            const pElement = document.createElement('div');
            pElement.textContent = inputElement.value;
            pElement.addEventListener('click', () => changeInput(pElement.textContent))
            historyElement.appendChild(pElement);
            
        }
    }catch (error){
        console.log(error)
    }
}



submitButton.addEventListener('click', getMessage)



function clearInput() {
    inputElement.value = '';
}




buttonNewChatElement.addEventListener('click', clearInput)