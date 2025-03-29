import random

# Sample recipe database (in a real app, you'd use an API or a larger database)
RECIPES = [
    {
        "name": "Apple Cinnamon Oatmeal",
        "ingredients": ["apple", "oats", "cinnamon", "milk", "honey"],
        "instructions": "Dice the apple. Cook oats with milk. Add diced apple, cinnamon, and honey.",
        "healthy_score": 9
    },
    {
        "name": "Vegetable Stir Fry",
        "ingredients": ["carrot", "broccoli", "bell pepper", "onion", "garlic", "soy sauce", "rice"],
        "instructions": "Chop all vegetables. Stir fry with garlic. Add soy sauce. Serve over rice.",
        "healthy_score": 8
    },
    {
        "name": "Fruit Salad",
        "ingredients": ["apple", "banana", "orange", "grapes", "honey", "lemon juice"],
        "instructions": "Dice all fruits. Mix with honey and lemon juice.",
        "healthy_score": 10
    },
    {
        "name": "Roasted Vegetables",
        "ingredients": ["potato", "carrot", "onion", "bell pepper", "olive oil", "salt", "pepper", "rosemary"],
        "instructions": "Chop vegetables. Toss with oil and seasonings. Roast at 400°F for 30 minutes.",
        "healthy_score": 9
    },
    {
        "name": "Banana Smoothie",
        "ingredients": ["banana", "milk", "yogurt", "honey", "cinnamon"],
        "instructions": "Blend all ingredients until smooth.",
        "healthy_score": 8
    }
]

def recommend_recipes(main_ingredient, available_ingredients, is_fresh=True):
    """
    Recommend recipes based on the main ingredient and other available ingredients
    
    Args:
        main_ingredient: The main ingredient (detected from the image)
        available_ingredients: List of other ingredients the user has
        is_fresh: Whether the main ingredient is fresh or not
        
    Returns:
        list: Recommended recipes
    """
    # Combine all ingredients
    all_ingredients = [main_ingredient] + available_ingredients
    
    # Filter recipes that can be made with the available ingredients
    possible_recipes = []
    
    for recipe in RECIPES:
        # Count how many required ingredients are available
        required_ingredients = recipe["ingredients"]
        available_count = sum(1 for ing in required_ingredients if any(available_ing in ing or ing in available_ing for available_ing in all_ingredients))
        
        # Calculate match percentage
        match_percentage = (available_count / len(required_ingredients)) * 100
        
        # If we have at least 60% of the ingredients, consider this recipe
        if match_percentage >= 60:
            recipe_copy = recipe.copy()
            recipe_copy["match_percentage"] = match_percentage
            possible_recipes.append(recipe_copy)
    
    # Sort by match percentage and healthy score
    recommended_recipes = sorted(
        possible_recipes, 
        key=lambda x: (x["match_percentage"], x["healthy_score"]), 
        reverse=True
    )
    
    # If the ingredient is not fresh, prioritize cooked recipes
    if not is_fresh and len(recommended_recipes) > 1:
        # This is a simplified approach - in a real app, you'd have more sophisticated logic
        cooked_recipes = [r for r in recommended_recipes if "roast" in r["instructions"].lower() or "cook" in r["instructions"].lower()]
        if cooked_recipes:
            recommended_recipes = cooked_recipes + [r for r in recommended_recipes if r not in cooked_recipes]
    
    return recommended_recipes[:3]  # Return top 3 recommendations
