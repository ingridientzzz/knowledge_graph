# 🤖 Using Different Models in ChatDBT

ChatDBT can use **any model loaded in LM Studio**, not just the default `gpt-oss`!

## 🔍 **Step 1: Discover Available Models**

Run this script to see all models currently loaded in LM Studio:

```bash
python3 get_models.py
```

This will show output like:
```
📋 Available models in LM Studio:
==================================================
1. Model ID: llama-3.2-3b-instruct
   Type: model
   Owner: huggingface

2. Model ID: phi-3.5-mini-instruct  
   Type: model
   Owner: microsoft

3. Model ID: qwen2.5-7b-instruct
   Type: model
   Owner: alibaba
```

## ⚙️ **Step 2: Configure Your Model**

### **Option A: Environment Variable**
```bash
export LM_STUDIO_MODEL="llama-3.2-3b-instruct"
```

### **Option B: Update .env File**
Create/edit `.env` file in the backend directory:
```bash
LM_STUDIO_MODEL=llama-3.2-3b-instruct
```

### **Option C: Use Different Models for Different Queries**
You can even change models on the fly by updating the environment variable and restarting the backend.

## 🔄 **Step 3: Restart Backend**

After changing the model configuration:
```bash
# Stop current backend (Ctrl+C)
# Then restart:
source /Users/marquein/knowledge_graph/.venv/bin/activate
python3 main.py
```

## 🎯 **Popular Model Examples**

| Model Name | Best For | Memory |
|------------|----------|---------|
| `llama-3.2-3b-instruct` | Fast responses, good reasoning | ~6GB |
| `phi-3.5-mini-instruct` | Efficient, Microsoft-optimized | ~4GB |
| `qwen2.5-7b-instruct` | Multilingual, code generation | ~14GB |
| `mistral-7b-instruct-v0.3` | Balanced performance | ~14GB |
| `codellama-7b-instruct` | Code-focused tasks | ~14GB |

## 🔧 **Model-Specific Optimizations**

The ChatDBT query router will automatically adapt to different models:

- **Smaller models** (3B-7B): Get fewer context documents for faster processing
- **Larger models** (13B+): Can handle more context for comprehensive answers
- **Code-focused models**: Get optimized prompts for SQL and transformation queries

## 💡 **Tips**

1. **Load models in LM Studio first** before configuring ChatDBT
2. **Use exact model IDs** as shown by `get_models.py`
3. **Test with simple queries** after switching models
4. **Monitor memory usage** - larger models need more RAM

Your ChatDBT system will now work with any model you have loaded! 🚀
