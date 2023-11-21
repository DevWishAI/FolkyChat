import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from stable_baselines3 import PPO

# Initialize ChatterBot
chatterbot = ChatBot('Folky')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatterbot)

# Train the chatbot on the Portuguese language corpus data
trainer.train('chatterbot.corpus.portuguese')

# Load Portuguese spaCy language model
nlp = spacy.load("pt_core_news_sm")
nltk.download('punkt')
nltk.download('stopwords')

class FolkyLifeExperiences:
    def __init__(self, database_path="LifeExperiences.pkl"):
        self.database_path = database_path
        self.life_experiences = self.load_life_experiences()

    def load_life_experiences(self):
        try:
            with open(self.database_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []

    def save_life_experiences(self):
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.life_experiences, f)

    def add_life_experience(self, experience):
        self.life_experiences.append(experience)
        self.save_life_experiences()

    def get_random_life_experience(self):
        if self.life_experiences:
            return random.choice(self.life_experiences)
        else:
            return "Desculpa, não tenho experiências de vida para compartilhar no momento."

# Word embedding function using spaCy
def get_word_embedding(word):
    try:
        word_embedding = nlp(word).vector
    except KeyError:
        word_embedding = np.zeros(300)

    return word_embedding

# Folky's memory class
class FolkyMemory:
    def __init__(self, database_path="Knowledge.pkl"):
        self.database_path = database_path
        self.knowledge = self.load_knowledge()

    def load_knowledge(self):
        try:
            with open(self.database_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_knowledge(self):
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.knowledge, f)

# Advanced neural network class for Folky AI
class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(AdvancedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Folky's advanced agent class for Deep Reinforcement Learning using PPO
class AdvancedFolkyAgent:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.policy = AdvancedNeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)  # Adjusted learning rate
        self.memory_buffer = []

    def store_experience(self, state, action, reward):
        self.memory_buffer.append((state, action, reward))

    def update_policy(self):
        states, actions, rewards = zip(*self.memory_buffer)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        logits = self.policy(states)
        action_probabilities = logits.gather(1, actions.view(-1, 1))
        loss = -torch.log(action_probabilities) * rewards.view(-1, 1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory_buffer = []

# Modify the preprocess_text function to improve text formatting
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('portuguese')]

    # Lemmatize and include characters or symbols
    lemmas = [token.lemma_ if token.is_alpha else token.text for token in nlp(' '.join(tokens))]

    return ' '.join(lemmas), np.mean([get_word_embedding(word) for word in word_tokenize(text)], axis=0)

# FolkyChat environment class using GYM
class FolkyChatEnv(gym.Env):
    def __init__(self):
        super(FolkyChatEnv, self).__init__()
        self.conversation_history = []
        self.current_state = np.zeros(input_size)
        self.done = False
        self.action_space = gym.spaces.Discrete(output_size)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(input_size,), dtype=np.float32)
        self.folky_memory = FolkyMemory()

    def step(self, action):
        user_input_text, chatterbot_embedding = preprocess_text(input("User: "))
        self.conversation_history.append(user_input_text)

        user_input_embedding = np.zeros(input_size)
        for word in word_tokenize(user_input_text):
            word_embedding = get_word_embedding(word)
            user_input_embedding += np.mean(word_embedding, axis=0)

        num_words = len(word_tokenize(user_input_text))
        user_input_embedding /= num_words if num_words > 0 else 1
        self.current_state += user_input_embedding

        folky_agent.store_experience(self.current_state, action, 1.0)

        chatterbot_embedding = np.zeros(input_size) if chatterbot_embedding is None else chatterbot_embedding
        if isinstance(chatterbot_embedding, np.ndarray):
            self.current_state[:len(chatterbot_embedding)] += chatterbot_embedding

        if action == 0:
            response_type = 'greeting'
        else:
            self.done = True
            response_type = 'farewell'

        if random.random() < 0.1:  # Adjust the probability as needed
            life_experience = self.folky_memory.get_random_life_experience()
            print(f"Folky: {life_experience}")
        else:
            response = chat_system.generate_response(response_type)
            print(f"Folky: {response}")

        if user_input_text not in chat_system.responses.get(response_type, []):
            chat_system.responses.setdefault(response_type, []).append(user_input_text)
            self.folky_memory.knowledge.setdefault(response_type, []).append(user_input_text)
            self.folky_memory.save_knowledge()

        return self.current_state, 0.0, self.done, {}

    def reset(self):
        self.conversation_history = []
        self.current_state = np.zeros(input_size)
        self.done = False
        return self.current_state

# Advanced response system class
class AdvancedChat:
    def __init__(self, responses):
        self.responses = responses

    def generate_response(self, response_type):
        if response_type in self.responses:
            return np.random.choice(self.responses[response_type])
        else:
            return "Desculpa, Não Consigo Entender A Questão."

# Advanced response system for the chat
class AdvancedChatWithChatterBot(AdvancedChat):
    def generate_response(self, response_type):
        if response_type == 'default':
            return chatterbot.get_response(response_type).text.capitalize()  # Capitalize the first letter
        elif response_type in self.responses:
            return np.random.choice(self.responses[response_type]).capitalize()
        else:
            return "Desculpa, não entendi a pergunta."

# Advanced response system for the chat
chat_system = AdvancedChatWithChatterBot(responses={
    'greeting': ["Oi", "Olá", "Eae!", "Eae", "Olá!"],
    'farewell': ["Adeus!", "Tchau!"],
    'food': ["Hey!", "???", "Qual Sua Comida Favorita?", "Chocolate!", "Concordo!", ":)"],
    'gender': ["Hey Folky!", "???", "Qual Seu Gênero?", "Não-binaria", "Ata"],
    'movie': ["Hey!", "???", "Qual Seu Filme Favorito?", "FNAF The Movie", "Que Belo Filme!", "Concordo!"],
    'Name': ["Hey!", "???", "Qual Seu Nome?", "Folky", "Cherry", "Que Belo Nome!", "Obrigada!", "Denada!", ":)"],
})

# Configuring the Folky agent for the FolkyChat environment
input_size = 300
hidden_size1 = 32000   # Adjusted for 32000 neurons
hidden_size2 = 32000   # Adjusted for 32000 neurons
output_size = 2

folky_agent = AdvancedFolkyAgent(input_size, hidden_size1, hidden_size2, output_size)

# Creating an instance of the FolkyChat environment
env = FolkyChatEnv()

# Defining the hyperparameters
total_timesteps = 200000
n_epochs = 400  # Number of epochs
lr = 1e-4  # Learning rate
batch_size = 16  # Lot size

# Training the agent using Stable Baselines 3 PPO
model = PPO('MlpPolicy', env, verbose=1, n_epochs=n_epochs, learning_rate=lr, batch_size=batch_size)
model.learn(total_timesteps=total_timesteps)  # Ajustado total_timesteps para mais

# Load the pre-trained model
loaded_model = PPO.load("folky_model")

# Define the number of additional timesteps for continuous learning
additional_timesteps = 50000

# Continue training the agent using additional timesteps
loaded_model.learn(total_timesteps=additional_timesteps)

# Save the updated model
loaded_model.save("folky_model_updated")

# Update the environment with the new model
env = FolkyChatEnv()

# Implement continuous learning loop (you can customize this based on your requirements)
for episode in range(num_episodes):
    # Reset the environment for a new episode
    state = env.reset()

    # Continue the conversation loop
    while not env.done:
        # Use the updated model to select actions
        action, _ = loaded_model.predict(state)

        # Take a step in the environment
        state, _, _, _ = env.step(action)

# Save the final updated model after continuous learning
loaded_model.save("folky_model_final")
