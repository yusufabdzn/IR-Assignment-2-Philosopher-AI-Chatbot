from agent import Agent

agent = Agent()

print("AI Agent Started")
print("Type '/philosopher' to toggle Philosopher Mode\n")
print("Type '/quit' to exit")

# Track the state of philosopher mode
philosopher_mode = False

while True:
    # Visual indicator in the terminal prompt
    prompt_prefix = "You [Philosopher Mode]: " if philosopher_mode else "You: "
    user_input = input(prompt_prefix)

    if user_input.lower() == "/quit":
        break

    if user_input.lower() == "/philosopher":
        philosopher_mode = not philosopher_mode
        print(f"\n--- Philosopher Mode {'ENABLED' if philosopher_mode else 'DISABLED'} ---\n")
        continue

    # Pass the mode flag to the agent
    response = agent.chat(user_input, philosopher_mode=philosopher_mode)

    print("\nAgent:", response)
    print()