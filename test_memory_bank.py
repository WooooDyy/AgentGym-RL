"""
Simple test script for the Memory Bank functionality.

This script demonstrates:
1. Creating a memory bank
2. Adding experiences
3. Retrieving similar experiences
4. Saving and loading the memory bank

Run: python test_memory_bank.py
"""

import sys
import os
from verl.utils.memory import MemoryBank

# Add the AgentGym-RL directory to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AgentGym-RL'))



def test_basic_functionality() -> MemoryBank:
    """Test basic memory bank operations."""
    print("=" * 80)
    print("Test 1: Basic Memory Bank Functionality")
    print("=" * 80)

    # Create memory bank
    print("\n1. Creating memory bank...")
    memory_bank = MemoryBank(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # Use CPU for testing
        min_reward=0.5,
        task_specific=True,
    )
    print(f"   ✓ Memory bank created with encoder: {memory_bank.encoder_name}")
    print(f"   ✓ Embedding dimension: {memory_bank.embedding_dim}")

    # Add some sample experiences
    print("\n2. Adding sample experiences...")
    experiences = [
        ("You are in a kitchen. On the counter you see a thermometer. What do you do?", "take thermometer", 1.0, "sciworld", 1),
        ("You are holding a thermometer. There is a freezer here.", "open freezer", 0.8, "sciworld", 2),
        ("The task is to measure temperature. You have no tools.", "look around", 0.0, "sciworld", 3),  # Low reward, won't be stored
        ("You need to find substances with different melting points.", "examine substance", 0.9, "sciworld", 4),
        ("There is a beaker on the table.", "pick up beaker", 0.7, "sciworld", 5),
    ]

    stored_count = 0
    for obs, action, reward, task, item_id in experiences:
        was_stored = memory_bank.add(obs, action, reward, task, item_id)
        if was_stored:
            stored_count += 1
            print(f"   ✓ Stored: {obs[:50]}... (reward={reward})")
        else:
            print(f"   ✗ Filtered: {obs[:50]}... (reward={reward} < {memory_bank.min_reward})")

    print(f"\n   Total experiences stored: {len(memory_bank)} / {len(experiences)}")

    # Retrieve similar experiences
    print("\n3. Retrieving similar experiences...")
    query = "You encounter a locked door blocking your way."
    print(f"   Query: {query}")

    retrieved = memory_bank.retrieve(
        query_text=query,
        k=2,
        task_name="textcraft",
    )

    print(f"   Retrieved {len(retrieved)} similar experiences:")
    for i, exp in enumerate(retrieved, 1):
        print(f"\n   [{i}] Observation: {exp.obs_text}")
        print(f"       Action: {exp.action}")
        print(f"       Reward: {exp.reward}")

    return memory_bank


def test_chat_formatting():
    """Test formatting experiences as chat templates."""
    print("\n" + "=" * 80)
    print("Test 2: Chat Template Formatting")
    print("=" * 80)

    memory_bank = MemoryBank(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        min_reward=0.5,
        task_specific=True,
    )

    # Add experiences
    memory_bank.add("You see a thermometer on the table.", "pick up thermometer", 0.8, "sciworld", 1)
    memory_bank.add("There is a hot substance in the beaker.", "use thermometer on beaker", 0.9, "sciworld", 2)

    # Retrieve and format
    query = "You need to measure the temperature of a substance."
    retrieved = memory_bank.retrieve(query, k=2, task_name="sciworld")

    print("\n1. Retrieved experiences:")
    for exp in retrieved:
        print(f"   - {exp.obs_text} → {exp.action}")

    print("\n2. Formatted as chat template:")
    formatted = memory_bank.format_as_examples(retrieved, format_style='chat')
    for msg in formatted:
        print(f"   {msg['role']}: {msg['content']}")

    return memory_bank


def test_save_load():
    """Test saving and loading memory bank."""
    print("\n" + "=" * 80)
    print("Test 3: Save and Load")
    print("=" * 80)

    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "test_memory")

    try:
        # Create and populate memory bank
        print("\n1. Creating and populating memory bank...")
        memory_bank = MemoryBank(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            min_reward=0.5,
            task_specific=True,
        )

        memory_bank.add("Test observation 1", "test action 1", 0.8, "test_task", 1)
        memory_bank.add("Test observation 2", "test action 2", 0.9, "test_task", 2)
        print(f"   ✓ Added {len(memory_bank)} experiences")

        # Save
        print("\n2. Saving memory bank...")
        memory_bank.save(save_path)
        print(f"   ✓ Saved to {save_path}")
        print(f"   ✓ Files created: {save_path}.pkl, {save_path}.faiss")

        # Load
        print("\n3. Loading memory bank...")
        loaded_memory_bank = MemoryBank.load(save_path, device="cpu")
        print(f"   ✓ Loaded {len(loaded_memory_bank)} experiences")

        # Verify
        print("\n4. Verifying loaded data...")
        assert len(loaded_memory_bank) == len(memory_bank), "Experience count mismatch"
        assert loaded_memory_bank.encoder_name == memory_bank.encoder_name, "Encoder name mismatch"
        assert loaded_memory_bank.min_reward == memory_bank.min_reward, "Min reward mismatch"
        print("   ✓ All checks passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n5. Cleaned up temporary files")


def test_cross_task_retrieval():
    """Test cross-task retrieval."""
    print("\n" + "=" * 80)
    print("Test 4: Cross-Task Retrieval")
    print("=" * 80)

    # Create memory bank with cross-task retrieval enabled
    memory_bank = MemoryBank(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        min_reward=0.5,
        task_specific=False,  # Enable cross-task
    )

    # Add experiences from different tasks
    print("\n1. Adding experiences from different tasks...")
    memory_bank.add("Measure the temperature of the liquid", "use thermometer", 0.8, "sciworld_temperature", 1)
    memory_bank.add("Find substances with different melting points", "examine materials", 0.9, "sciworld_melting", 2)
    memory_bank.add("Identify the electrical conductor", "test with circuit", 0.7, "sciworld_conductivity", 3)
    print(f"   ✓ Added {len(memory_bank)} experiences from 3 different ScienceWorld tasks")

    # Retrieve across tasks
    print("\n2. Retrieving experiences (cross-task)...")
    query = "You need to determine the properties of a substance"
    retrieved = memory_bank.retrieve(query, k=2, task_name=None)

    print(f"   Query: {query}")
    print(f"   Retrieved {len(retrieved)} experiences:")
    for exp in retrieved:
        print(f"   - [{exp.task_name}] {exp.obs_text} → {exp.action}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Memory Bank Test Suite" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Run tests
        test_basic_functionality()
        test_chat_formatting()
        test_save_load()
        test_cross_task_retrieval()

        # Summary
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        print("\nThe memory bank is ready to use in agent training.")
        print("Enable it in your config with:")
        print("  rollout:")
        print("    memory:")
        print("      enabled: true")
        print("      k: 3")
        print("      min_reward: 0.5")
        print("      save_path: 'outputs/memory_bank/task_name'")
        print()

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
