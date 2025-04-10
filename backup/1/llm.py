import ollama
from ollama import chat

from voice import text_to_speech

def get_llm_response(prompt: str) -> str:

    response = chat(
        model = 'fotiecodes/jarvis:latest',  # Specify the model you want to use
        messages=[{"role": "user", "content": prompt}],  # The input message for the model
    )
    return response.message.content  # Extract the content of the response

if __name__ == '__main__':
    while True:

        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        # Get the LLM response
        llm_response = get_llm_response(user_input)
        print("Jarvis: {llm_response}")

        # Convert the LLM response to speech
        text_to_speech(llm_response)
