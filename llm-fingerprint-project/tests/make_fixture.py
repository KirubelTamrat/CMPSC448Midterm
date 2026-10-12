"""Make obviously synthetic responses for smoke-testing the pipeline ONLY."""
from pathlib import Path
import csv
cats=["General Knowledge", "Coding", "Mathematics", "Writing", "Reasoning", "Translation", "Safety"]
rows=[]
for c in cats:
 for i in range(6):
  question=f"[SYNTHETIC TEST PROMPT] {c}: example question number {i+1}."
  for name in ("ChatGPT","Claude","Gemini"):
   filler={"ChatGPT":"Sure. Here are three things to keep in mind: first, second, and third.",
           "Claude":"One way to approach this is to consider the important context and tradeoffs.",
           "Gemini":"## Overview\n* Main point one\n* Main point two\nIn summary, examples matter."}[name]
   rows.append({"prompt_id":f"demo_{c}_{i}","LLM_name":name,"LLM_Input":question,
                "LLM_output":"[SYNTHETIC TEST DATA ONLY, NOT AN ACTUAL LLM RESPONSE] "+filler+" "*3+filler,
                "category":c})
with Path(__file__).with_name("fixture_demo.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("Synthetic rows:",len(rows))
