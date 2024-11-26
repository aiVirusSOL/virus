import os
import json
from typing import Any, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class virusAI:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        print(f"Model {model_name} loaded successfully.")

    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """Generate a response from the model given a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def train(self, data_path: str, epochs: int = 1, batch_size: int = 4, lr: float = 5e-5):
        """A lightweight training pipeline for fine-tuning the model."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")
        
        print("Loading training data...")
        with open(data_path, "r") as file:
            data = json.load(file)
        
        texts = data["texts"]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        print("Starting fine-tuning...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        print("Fine-tuning complete. Model updated.")

    def save_model(self, save_path: str):
        """Save the fine-tuned model to disk."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}.")

    def load_model(self, load_path: str):
        """Load a fine-tuned model from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
        print(f"Model loaded from {load_path}.")

def main():
    # Example usage of virus AI
    print("Initializing virus AI...")
    virus_ai = virusAI()

    # Example: Generate response
    print("\nGenerating a response...")
    prompt = "What is the purpose of life?"
    response = virus_ai.generate_response(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}")

    # Example: Training (requires a valid dataset in JSON format)
    # dataset = {"texts": ["Example training text 1.", "Example training text 2."]}
    # Save it as a file: training_data.json
    # with open("training_data.json", "w") as f:
    #     json.dump(dataset, f)
    # virus_ai.train("training_data.json", epochs=1)

if __name__ == "__main__":
    main()
