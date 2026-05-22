import os
import shutil

def reset_agent_memory():
    print("🔄 Initializing memory reset sequence...")
    
    # 1. Wipe out the ChromaDB database directory
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print("✅ Successfully purged ChromaDB vector storage.")
        except Exception as e:
            print(f"❌ Error removing chroma_db directory: {e}")
    else:
        print("ℹ️ No ChromaDB directory found. Already clean.")

    # 2. Wipe out the consolidated summary text file
    summary_file = "summary.txt"
    if os.path.exists(summary_file):
        try:
            os.remove(summary_file)
            print("✅ Successfully purged long-term summary cache.")
        except Exception as e:
            print(f"❌ Error removing summary.txt: {e}")
    else:
        print("ℹ️ No summary.txt found. Already clean.")

    print("\n🚀 Reset complete! Start your agent for a completely fresh session.")

if __name__ == "__main__":
    reset_agent_memory()