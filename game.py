import gradio as gr
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
import os
import json
from datetime import datetime
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("adventure_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger("adventure_game")

load_dotenv()

try:
    chat_model = ChatMistralAI(
        temperature=0.7,
        max_tokens=500
    )
    logger.info("Chat model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat model: {str(e)}")
    raise

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.cache = {}
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                logger.info("Image model loaded on CUDA")
            else:
                logger.warning("CUDA not available, using CPU for image generation (will be slow)")
            self.pipe.enable_attention_slicing()
        except Exception as e:
            logger.error(f"Failed to initialize image generator: {str(e)}")
            self.pipe = None
    
    def generate(self, description, num_inference_steps=30, guidance_scale=7.5):
        cache_key = f"{description}_{num_inference_steps}_{guidance_scale}"
        if cache_key in self.cache:
            logger.info("Using cached image")
            return self.cache[cache_key]
        if not self.pipe:
            logger.error("Image generator not properly initialized")
            return None
        try:
            styled_prompt = f"Cinematic view, fantasy adventure, child-friendly, vibrant colors, detailed illustration style. {description}"
            logger.info(f"Generating image for prompt: {styled_prompt[:50]}...")
            start_time = time.time()
            image = self.pipe(
                styled_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            generation_time = time.time() - start_time
            logger.info(f"Image generated in {generation_time:.2f} seconds")
            self.cache[cache_key] = image
            if len(self.cache) > 20:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            return image
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None

image_generator = ImageGenerator()

class AdventureGame:
    def __init__(self):
        self.history = []
        self.current_story_id = self._generate_story_id()
        self.save_dir = "adventure_saves"
        self._ensure_save_directory()
        
    def _generate_story_id(self):
        return f"adventure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _ensure_save_directory(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Created save directory: {self.save_dir}")
    
    def get_story_context(self, max_history=5):
        recent_history = self.history[-max_history:] if len(self.history) > max_history else self.history
        context = "\n".join([
            f"USER: {msg['content']}" if msg['role'] == 'user' else f"STORY: {msg['content']}"
            for msg in recent_history
        ])
        return context
    
    def get_response(self, user_input, system_prompt=None):
        if not system_prompt:
            system_prompt = """You are an AI that generates engaging text-adventure storylines for kids ages 8-12.
            Follow these rules:
            1. Create detailed, immersive scenes that respond directly to the user's input
            2. Maintain story coherence based on previous interactions
            3. Include choices or decision points at the end of most responses
            4. Keep content appropriate for children (no violence, scary content, or adult themes)
            5. If the story reaches a natural conclusion, use 'THE END' at the very end
            6. Keep responses concise (3-4 paragraphs maximum)
            7. Include sensory details that would make for good visualizations
            """
        try:
            messages = [
                SystemMessage(content=system_prompt),
            ]
            context = self.get_story_context(5)
            messages.append(HumanMessage(content=f"Previous story context:\n{context}\n\nUser's next action: {user_input}"))
            response = chat_model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return "Sorry, I'm having trouble continuing the story right now. Please try again."
    
    def summarize_for_image(self, text):
        try:
            messages = [
                SystemMessage(content="Generate a detailed visual description in less than 60 words that captures the most visual and important elements of this scene for an illustration. Focus on characters, setting, and action."),
                HumanMessage(content=text)
            ]
            summary_response = chat_model.invoke(messages)
            logger.info(f"Image prompt generated: {summary_response.content}")
            return summary_response.content
        except Exception as e:
            logger.error(f"Error generating image prompt: {str(e)}")
            return "Fantasy adventure scene with characters"
    
    def save_state(self):
        save_path = os.path.join(self.save_dir, f"{self.current_story_id}.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'story_id': self.current_story_id,
                    'timestamp': datetime.now().isoformat(),
                    'history': self.history
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Game state saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving game state: {str(e)}")
            return False
    
    def load_state(self, story_id):
        save_path = os.path.join(self.save_dir, f"{story_id}.json")
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data['history']
                    self.current_story_id = data['story_id']
                logger.info(f"Game state loaded from {save_path}")
                return True
            else:
                logger.warning(f"Save file not found: {save_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading game state: {str(e)}")
            return False
    
    def list_saved_games(self):
        try:
            saves = []
            for filename in os.listdir(self.save_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self.save_dir, filename), 'r') as f:
                            data = json.load(f)
                            saves.append({
                                'id': data['story_id'],
                                'timestamp': data['timestamp'],
                                'preview': data['history'][1]['content'][:100] + "..." if len(data['history']) > 1 else "New adventure"
                            })
                    except:
                        pass
            return sorted(saves, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            logger.error(f"Error listing saved games: {str(e)}")
            return []
    
    def reset(self):
        self.history = []
        self.current_story_id = self._generate_story_id()
        logger.info("Game reset")
    
    def add_to_history(self, role, content):
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def game_logic(self, user_input):
        if user_input.lower().strip() == "\\reset":
            self.reset()
            return "Adventure reset! Type something to begin a new story.", None
        if user_input.lower().strip() == "\\save":
            self.save_state()
            return "Game saved! Continue your adventure or type \\load to see saved games.", None
        if user_input.lower().strip() == "\\load":
            saves = self.list_saved_games()
            if saves:
                save_list = "\n".join([f"{i+1}. {save['timestamp']} - {save['preview']}" for i, save in enumerate(saves[:5])])
                return f"Saved adventures:\n{save_list}\n\nType \\load:NUMBER to load a game (e.g., \\load:1)", None
            else:
                return "No saved adventures found. Start a new one!", None
        if user_input.lower().startswith("\\load:"):
            try:
                index = int(user_input.split(":")[1]) - 1
                saves = self.list_saved_games()
                if 0 <= index < len(saves):
                    self.load_state(saves[index]['id'])
                    last_ai_response = next((msg['content'] for msg in reversed(self.history) 
                                            if msg['role'] == 'assistant'), "Adventure loaded!")
                    image_prompt = self.summarize_for_image(last_ai_response)
                    image = image_generator.generate(image_prompt)
                    return f"Adventure loaded! Here's where you left off:\n\n{last_ai_response}", image
                else:
                    return "Invalid save number. Try again or type \\load to see the list.", None
            except Exception as e:
                logger.error(f"Error loading game: {str(e)}")
                return "Error loading game. Please try again.", None
        self.add_to_history('user', user_input)
        ai_response = self.get_response(user_input)
        self.add_to_history('assistant', ai_response)
        self.save_state()
        if "THE END" in ai_response.upper():
            ai_response += "\n\nThis adventure has concluded. Type anything to start a new story."
            self.reset()
        image_prompt = self.summarize_for_image(ai_response)
        image = image_generator.generate(image_prompt)
        return ai_response, image
    
    def initialize_game(self):
        if not self.history:
            init_prompt = "Begin a new fantasy adventure for a young explorer."
            self.add_to_history('user', init_prompt)
            ai_response = self.get_response(init_prompt)
            self.add_to_history('assistant', ai_response)
            self.save_state()
            image_prompt = self.summarize_for_image(ai_response)
            image = image_generator.generate(image_prompt)
            return ai_response, image
        else:
            last_response = next((msg['content'] for msg in reversed(self.history) 
                                  if msg['role'] == 'assistant'), "Begin your adventure!")
            image_prompt = self.summarize_for_image(last_response)
            image = image_generator.generate(image_prompt)
            return last_response, image

game = AdventureGame()

with gr.Blocks(title="AI-Powered Text Adventure for Kids") as demo:
    gr.Markdown("""
    # 🧙‍♂️ AI-Powered Text Adventure for Kids 🗺️
    
    Type your actions to play! Special commands:
    - \\reset - Start a new adventure
    - \\save - Save your current adventure
    - \\load - View your saved adventures
    """)
    with gr.Row():
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="Story", 
                lines=10,
                elem_id="story-output"
            )
            user_input = gr.Textbox(
                label="What do you want to do?", 
                placeholder="Type your action here...",
                elem_id="user-input"
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("New Adventure")
                save_btn = gr.Button("Save Game")
                load_btn = gr.Button("Load Game")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Scene", elem_id="scene-image")
            with gr.Accordion("Game Info", open=False):
                game_info = gr.Markdown("""
                ## How to Play
                
                1. Read the story text
                2. Type what you want your character to do
                3. Press Submit or Enter
                4. See what happens next!
                
                This is an interactive story where YOU decide what happens.
                """)
    submit_event = user_input.submit(
        game.game_logic, 
        inputs=user_input, 
        outputs=[output_text, output_image]
    ).then(lambda: "", None, user_input)
    submit_btn.click(
        game.game_logic, 
        inputs=user_input, 
        outputs=[output_text, output_image]
    ).then(lambda: "", None, user_input)
    reset_btn.click(
        lambda: game.reset() or ("Adventure reset! Type something to begin.", None),
        inputs=None,
        outputs=[output_text, output_image]
    )
    save_btn.click(
        lambda: game.save_state() and "Game saved! Continue your adventure or load a different one.",
        inputs=None,
        outputs=output_text
    )
    load_btn.click(
        lambda: (game.list_saved_games(), None) if game.list_saved_games() else ("No saved games found.", None),
        inputs=None,
        outputs=[output_text, output_image]
    )
    demo.load(game.initialize_game, outputs=[output_text, output_image])

if __name__ == "__main__":
    try:
        demo.launch(
            share=True,
            debug=False,
            server_port=7860
        )
    except Exception as e:
        logger.critical(f"Failed to launch application: {str(e)}")