css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://images.aeonmedia.co/images/6041fd2f-991e-4f4f-8895-28886f3a5071/original.jpg?width=3840&quality=75&format=auto" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://img.freepik.com/free-vector/cute-cat-confused-cartoon-vector-icon-illustration-animal-nature-icon-concept-isolated-flat-vector_138676-9619.jpg?size=338&ext=jpg&ga=GA1.1.2082370165.1716422400&semt=ais_user">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''