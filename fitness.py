import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
from datetime import datetime
import json

class FitnessAI:
    def __init__(self, model_path, user_profile_path=None):
        """
        Initialize the Fitness AI system with LLaMA model and user profile
        
        Args:
            model_path: Path to the LLaMA model weights
            user_profile_path: Optional path to load existing user profile
        """
        # Initialize LLaMA model and tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Initialize user profile
        self.user_profile = self._load_user_profile(user_profile_path) if user_profile_path else {
            "fitness_level": 0,
            "exercise_history": [],
            "performance_metrics": {},
            "preferences": {},
        }
        
        # Performance thresholds for different fitness levels
        self.fitness_thresholds = {
            "beginner": {"heart_rate": 140, "form_accuracy": 0.6},
            "intermediate": {"heart_rate": 160, "form_accuracy": 0.75},
            "advanced": {"heart_rate": 180, "form_accuracy": 0.9},
        }

    def process_real_time_data(self, sensor_data):
        """
        Process real-time sensor data and generate workout adjustments
        
        Args:
            sensor_data: Dict containing real-time metrics (heart_rate, form_data, etc.)
        Returns:
            Dict containing workout adjustments and recommendations
        """
        # Prepare context for LLaMA
        context = self._prepare_context(sensor_data)
        
        # Generate recommendations using LLaMA
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=200,
                temperature=0.7,
                num_return_sequences=1,
            )
        
        recommendations = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse and structure the recommendations
        adjustments = self._parse_recommendations(recommendations)
        
        # Update user profile with new data
        self._update_user_profile(sensor_data, adjustments)
        
        return adjustments

    def _prepare_context(self, sensor_data):
        """
        Prepare context string for LLaMA model
        """
        context = f"""
        User Fitness Level: {self.user_profile['fitness_level']}
        Current Heart Rate: {sensor_data.get('heart_rate', 0)} BPM
        Form Accuracy: {sensor_data.get('form_accuracy', 0):.2f}
        Exercise: {sensor_data.get('exercise_type', 'unknown')}
        
        Based on the above metrics, suggest real-time adjustments for:
        1. Exercise intensity
        2. Form corrections
        3. Rest periods
        4. Progressive overload
        """
        return context

    def _parse_recommendations(self, llama_output):
        """
        Parse LLaMA output into structured recommendations
        """
        try:
            # Basic parsing of LLaMA output into categories
            adjustments = {
                "intensity_adjustment": None,
                "form_corrections": [],
                "rest_period": None,
                "progression_advice": None,
            }
            
            # Split output into lines and categorize recommendations
            lines = llama_output.split('\n')
            current_category = None
            
            for line in lines:
                if "intensity" in line.lower():
                    adjustments["intensity_adjustment"] = line.split(":")[-1].strip()
                elif "form" in line.lower():
                    adjustments["form_corrections"].append(line.split(":")[-1].strip())
                elif "rest" in line.lower():
                    adjustments["rest_period"] = line.split(":")[-1].strip()
                elif "progress" in line.lower():
                    adjustments["progression_advice"] = line.split(":")[-1].strip()
            
            return adjustments
            
        except Exception as e:
            print(f"Error parsing recommendations: {e}")
            return {"error": "Failed to parse recommendations"}

    def _update_user_profile(self, sensor_data, adjustments):
        """
        Update user profile with new exercise data and adjustments
        """
        timestamp = datetime.now().isoformat()
        
        # Record exercise session
        session_data = {
            "timestamp": timestamp,
            "sensor_data": sensor_data,
            "adjustments": adjustments,
        }
        
        self.user_profile["exercise_history"].append(session_data)
        
        # Update performance metrics
        if "heart_rate" in sensor_data:
            if "heart_rate" not in self.user_profile["performance_metrics"]:
                self.user_profile["performance_metrics"]["heart_rate"] = []
            self.user_profile["performance_metrics"]["heart_rate"].append({
                "timestamp": timestamp,
                "value": sensor_data["heart_rate"]
            })

        # Adjust fitness level based on performance
        self._adjust_fitness_level(sensor_data)

    def _adjust_fitness_level(self, sensor_data):
        """
        Adjust user's fitness level based on performance metrics
        """
        current_level = self.user_profile["fitness_level"]
        
        # Check performance against thresholds
        hr = sensor_data.get("heart_rate", 0)
        form = sensor_data.get("form_accuracy", 0)
        
        # Simple logic for fitness level adjustment
        if (hr > self.fitness_thresholds["advanced"]["heart_rate"] and 
            form > self.fitness_thresholds["advanced"]["form_accuracy"]):
            new_level = min(current_level + 1, 10)  # Cap at level 10
        elif (hr < self.fitness_thresholds["beginner"]["heart_rate"] or 
              form < self.fitness_thresholds["beginner"]["form_accuracy"]):
            new_level = max(current_level - 1, 0)  # Minimum level 0
        else:
            new_level = current_level
            
        self.user_profile["fitness_level"] = new_level

    def _load_user_profile(self, profile_path):
        """
        Load user profile from JSON file
        """
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading user profile: {e}")
            return None

    def save_user_profile(self, profile_path):
        """
        Save current user profile to JSON file
        """
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving user profile: {e}")
            return False