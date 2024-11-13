const bot = "{% static 'img/bot.svg' %}"
const user = "{% static 'img/user.svg' %}"
const submitButton = document.querySelector('#submit');
const inputElement = document.querySelector('textarea');
const historyElement = document.querySelector('.history');
const buttonNewChatElement = document.querySelector('.new-chat');
const form = document.querySelector('form');
const whatContainer = document.querySelector('.what-container');

const chatContainer = document.querySelector('.chat-container')

let session_messages;

const chat_sessions = JSON.parse(document.getElementById('chat_sessions').textContent);

function isSameDay(d1, d2) {
    return d1.getFullYear() === d2.getFullYear() &&
        d1.getMonth() === d2.getMonth() &&
        d1.getDate() === d2.getDate();
}

const today = new Date();
const yesterday = new Date(today);
yesterday.setDate(today.getDate() - 1);
const last7Days = new Date(today);
last7Days.setDate(today.getDate() - 7);
const last30Days = new Date(today);
last30Days.setDate(today.getDate() - 30);





function updateChatSession(chat_sessions) {

    chat_sessions.forEach(session => {

        const createdAt = new Date(session.created_at);
        let groupElement;

        if (isSameDay(createdAt, today)) {
            groupElement = document.getElementById('today').querySelector('.chat-list');
        } else if (isSameDay(createdAt, yesterday)) {
            groupElement = document.getElementById('yesterday').querySelector('.chat-list');
        } else if (createdAt >= last7Days) {
            groupElement = document.getElementById('last-7-days').querySelector('.chat-list');
        } else if (createdAt >= last30Days) {
            groupElement = document.getElementById('last-30-days').querySelector('.chat-list');
        }



        if (groupElement) {
            const existingChatItem = groupElement.querySelector(`a[href='/my_app/ai/chatbotpage/${session.chat_session_id}']`);
            if (!existingChatItem) {
                const chatItem = document.createElement('li');
                const chatDiv = document.createElement('div');
                chatItem.classList.add('chat-item')

                const chatLink = document.createElement('a');
                chatLink.style = "text-decoration: none;"
                chatLink.href = `/my_app/ai/chatbotpage/${session.chat_session_id}`;

                chatItem.appendChild(chatDiv);
                chatDiv.appendChild(chatLink);
                chatLink.textContent = `${session.session_title}`;
                groupElement.prepend(chatItem);
            }

        }
    });
}


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


document.addEventListener("DOMContentLoaded", () => {
    const whatCanIHelp = document.querySelector('.what-can-i-help')
    const circle = document.getElementById("circle-icon");



    function loadWhatCanIHelp() {
        whatCanIHelp.innerHTML = ''
        let text = "What can I help with?"
        let index = 0

        let interval = setInterval(() => {
            if (index < text.length) {
                whatCanIHelp.innerHTML += text.charAt(index)
                index++

                circle.style.left = `${index * 18}px`;
            } else {
                clearInterval(interval);

                let op = 1.0;
                let fadeInterval = setInterval(() => {
                    if (op >= 0) {
                        circle.style.opacity = op;
                        op -= 0.2;
                    } else {
                        clearInterval(fadeInterval);
                    }
                }, 40)
                setTimeout(() => circle.remove(), 1000);
            }
        }, 20)

    }

    loadWhatCanIHelp();


    const chat_sessions_two = JSON.parse(document.getElementById('chat_sessions').textContent);

    const curr_chat_session = JSON.parse(document.getElementById('curr_chat_session').textContent);


    updateChatSession(chat_sessions_two);


    if (curr_chat_session != null) {
        const isElementExist = document.querySelector('.what-container') !== null;
        if (isElementExist) {
            whatContainer.remove();
        }

        let messages = curr_chat_session['messages']
        for (let i = 0; i < messages.length; i++) {
            const message = messages[i];
            console.log("message: ", message)
            const uniqueId = generateUniqueId();

            let userPrompt = message['prompt_content'];

        

            let aiPrompt = message['chatbot_content'];

           

            chatContainer.innerHTML += chatStripe(false, userPrompt, uniqueId, '');
            chatContainer.innerHTML += chatStripe(true, aiPrompt, uniqueId, message['context']);
        }
    };

    let chatList = document.querySelectorAll('.chat-item');

    const currentUrl = window.location.pathname;
    chatList.forEach(function(item) {
        const link = item.querySelector('a');
        if (link && currentUrl === link.getAttribute('href')) {
            item.classList.add('active');
        }
    });

    session_messages = document.querySelectorAll('div.message');



});





function loader(element) {
    element.textContent = ''

    loadInterval = setInterval(() => {
        element.textContent += '.';

        if (element.textContent === '........') {
            element.textContent = '';
        }
    }, 300);
}

function typeText(element, text) {
    let index = 0
    // kalau pakai ini jawaban chatbotnya gak urut

    // let lines = text.split('\n');

    // let uniqueLines = [...new Set(lines)];

    // text = uniqueLines.join('\n');


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


function chatStripe(isAi, value, uniqueId, context) {
    if (isAi) {
        return (
            `
            <div class="wrapper ai">
                <div class="chat">
                    <div class="profile">
                        <img 

                          src="/static/img/bot.svg"
                          alt="bot"
                        />
                    </div>
                    <div class="message" id=${uniqueId}>${value}</div>
                     <a class="context_button" href="#context-${uniqueId}" id="context-${uniqueId}-c">See Context/Reference</a>
                     
                     <div class="modal_container" id="context-${uniqueId}"  >
                        <div class="modal">
                                <h2 class="modal_title">Context/Reference</h2>
                                <p class="modal_text" id="text-context-${uniqueId}">${context} </p>
                                <a href="#context-${uniqueId}-c" class="context_closer"> </a>
                        </div>
                     </div>
                </div>
            </div>
        `
        )
    } else {
        return (
            `
            <div class="wrapper  user">
                <div class="chat">
                    
                    <div class="message" id=${uniqueId}>${value}</div>
                    <div class="profile">
                        <img 
                        
                          src="/static/img/user.svg"
                          alt="user"
                        />
                    </div>
                </div>
            </div>
        `
        )
    }

}

let chatHistory = []



const csrftoken = getCookie('csrftoken');

function isValidUUID(uuid) {
    const regex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    return regex.test(uuid);
}

async function getMessage(e) {
    const isElementExist = document.querySelector('.what-container') !== null;
    if (isElementExist) {
        whatContainer.remove();
    }
    e.preventDefault();


    const data = new FormData(form);
    const uniqueIdUser = generateUniqueId()
    chatContainer.innerHTML += chatStripe(false, data.get('prompt'), uniqueIdUser, "")


    const uniqueId = generateUniqueId()
    chatContainer.innerHTML += chatStripe(true, " ", uniqueId, "")

    chatContainer.scrollTop = chatContainer.scrollHeight;

    const messageDiv = document.getElementById(uniqueId)
    const contextDiv = document.getElementById(`text-context-${uniqueId}`)

    loader(messageDiv)

    for (let i = 0; i < session_messages.length; i += 2) {
        let sess_message_user = session_messages[i];
        let sess_message_ai = session_messages[i + 1];
        chatHistory.push({
            'user': sess_message_user.textContent,
            'ai': sess_message_ai.textContent,
        });
    }



    context = ""
    let body = inputElement.value
    if (chatHistory.length != 0) {
        for (let i = 0; i < chatHistory.length; i++) {
            context += `
            user: ${chatHistory[i]['user'] }\
            n `
            context += `
            ai: ${chatHistory[i]['ai']}\
            n `
        }
    }

    const pathSegments = window.location.pathname.split('/');
    const param = pathSegments[pathSegments.length - 1];
    console.log("param: ", param)
    let chat_uuid = 0;
    if (isValidUUID(param)) {
        chat_uuid = param;
    }

    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken
        },
        body: JSON.stringify({
            question: body,
            chatHistory: context,
            chatUUID: chat_uuid
        })
    }



    try {

        const response = await fetch("http://localhost:8000/my_app/ai/chatbot", options)


        clearInterval(loadInterval)
        messageDiv.innerHTML = " "

        let data;
        if (response.ok) {
            data = await response.json();
            const parsedData = data.chatbot_message.trim()

            if (data.new_session_id != 0) {
                console.log("data res: ", data)

                const new_session_id = data.new_session_id

                chat_sessions.push({
                    chat_session_id: new_session_id,
                    created_at: data.new_session_created_at,
                    session_title: data.new_session_title
                });
                updateChatSession(chat_sessions);
            }
            typeText(messageDiv, parsedData)

            const context = data.context.trim()

            contextDiv.innerHTML = context



            chatHistory[chatHistory.length - 1]['ai'] = parsedData
        } else {
            const err = await response.text()

            messageDiv.innerHTML = "Something went wrong"
            alert("Gemini is not available at the moment  (usage limit reached). Please try again later in 1-2 minutes.")
        }



        if (data.chatbot_message) {
            const pElement = document.createElement('div');
            pElement.textContent = inputElement.value;
            pElement.addEventListener('click', () => changeInput(pElement.textContent))
            historyElement.appendChild(pElement);
        }
        inputElement.value = '';
    } catch (error) {
        console.log(error)
    }
}



submitButton.addEventListener('click', getMessage)



function clearChat() {
    let pathnameWithoutUUID = window.location.pathname

    if (window.location.pathname.split('/').length == 5) {
        pathnameWithoutUUID = pathnameWithoutUUID.split('/').slice(0, -1).join('/');
    }
    window.location.href = window.location.origin + pathnameWithoutUUID;
    inputElement.value = '';
    chatContainer.innerHTML = '';
}




buttonNewChatElement.addEventListener('click', clearChat)


// # kalau pakai ini jawaban chatbotnya gak urut

 // let userPrompt = message['prompt_content'].split('\n');

            // let uniqueLinesUser = [...new Set(userPrompt)];

            // userPrompt = uniqueLinesUser.join('\n');

            // let aiPrompt = message['chatbot_content'].split('\n');

            // let uniqueLinesAI = [...new Set(aiPrompt)];

            // aiPrompt = uniqueLinesAI.join('\n');