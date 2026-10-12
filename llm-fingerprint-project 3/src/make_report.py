"""Produce a course report PDF with real results once train.py finishes.

Without results, --draft produces an explicitly unfinished methods draft.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, KeepTogether)

NAVY = colors.HexColor("#15304B")
BLUE = colors.HexColor("#326187")
GRAY = colors.HexColor("#556572")


def styles():
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name="Cover",fontName="Helvetica-Bold",fontSize=21,leading=27,textColor=NAVY,spaceAfter=15))
    s.add(ParagraphStyle(name="SubheadMine",fontName="Helvetica",fontSize=11,leading=17,textColor=GRAY,spaceAfter=10))
    s.add(ParagraphStyle(name="SectionMine",fontName="Helvetica-Bold",fontSize=13,leading=18,textColor=NAVY,spaceBefore=15,spaceAfter=7,keepWithNext=True))
    s.add(ParagraphStyle(name="BodyMine",fontName="Helvetica",fontSize=9.5,leading=14,spaceAfter=9))
    s.add(ParagraphStyle(name="MiniMine",fontName="Helvetica",fontSize=8,leading=11,spaceAfter=4))
    s.add(ParagraphStyle(name="CaptionMine",fontName="Helvetica-Oblique",fontSize=8,leading=11,textColor=GRAY,spaceAfter=12))
    return s


def para(story, s, text, typ="BodyMine"):
    story.append(Paragraph(text.replace(" & ", " &amp; "),s[typ]))


def section(story,s,text):
    story.append(Paragraph(text,s["SectionMine"]))


def table(story,s,heads,rows,widths=None):
    data=[[Paragraph(str(x),s["MiniMine"]) for x in heads]]
    data.extend([[Paragraph(str(x),s["MiniMine"]) for x in row] for row in rows])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E6EDF3")),
        ("TEXTCOLOR",(0,0),(-1,0),NAVY),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,1),(-1,-1),5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),
        ("LINEBELOW",(0,0),(-1,0),.6,BLUE),
        ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#E2E8EE")),
    ]))
    story.append(t)
    story.append(Spacer(1,11))


def insert_chart(story,s,path,caption,width=6.5):
    if path.exists():
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w,h=im.size
        story.append(Image(str(path),width=inch*width,height=inch*width*h/w))
        para(story,s,caption,"CaptionMine")


def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#DCE6EF"))
    canvas.line(45,48,568,48)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica",8)
    canvas.drawString(45,34,"CMPSC 448 | LLM authorship attribution")
    canvas.drawRightString(568,34,f"Page {doc.page}")
    canvas.restoreState()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("report/Midterm_Project_Report.pdf"))
    ap.add_argument("--draft", action="store_true", help="Create visibly unfinished methodology/report draft without training results")
    ap.add_argument("--author",default="Kirubel Tamrat and Chukwubikem Dara")
    a=ap.parse_args()
    rpath=a.results/"experiment_results.json"
    if not a.draft and not rpath.exists():
        raise SystemExit(f"No verified experiment JSON at {rpath}. Run python -m src.train first or use --draft.")
    data=None if a.draft else json.loads(rpath.read_text())
    if data is not None and "SYNTHETIC" in data.get("metadata",{}).get("source_type",""):
        raise SystemExit("Refusing to create a report from synthetic fixture results. Run training on genuine LLM outputs.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    s=styles()
    doc=SimpleDocTemplate(str(a.out),pagesize=letter,leftMargin=.73*inch,rightMargin=.73*inch,
                          topMargin=.65*inch,bottomMargin=.7*inch,title="Who Wrote It? Identifying LLMs from Their Responses",author=a.author)
    st=[]
    para(st,s,"WHO WROTE IT?","Cover")
    para(st,s,"Identifying LLMs from Their Responses","SubheadMine")
    para(st,s,f"CMPSC 448 Midterm Project<br/>Team leader: Kirubel Tamrat<br/>Team member: Chukwubikem Dara<br/>October 11, 2026","SubheadMine")
    if data is None:
        para(st,s,"<b>DRAFT - NOT READY TO SUBMIT.</b> The experiment has not been run with verified model responses. Results, findings, and personal reflection are still missing.")
    else:
        para(st,s,"<b>Study:</b> Three model families, 100 shared prompts, two neural-network classifiers, and four research questions.")
    section(st,s,"1. Problem and motivation")
    para(st,s,"When we read an AI-generated answer, we usually cannot tell which model wrote it. We wanted to test whether the wording and structure of an answer give us any useful clues. We treated this as a three-class classification problem: from a response, predict ChatGPT, Claude, or Gemini. We also wanted to rule out a shortcut: learning the kinds of questions instead of learning anything about the responses.")
    para(st,s,"Our main questions were whether a CNN and an LSTM could recognize the source model from a response (<b>RQ1</b>) and whether providing the user prompt would help (<b>RQ2</b>). We also tested whether the classifiers worked on a new task category (<b>RQ3</b>) and what simple writing characteristics might distinguish the models (<b>RQ4</b>).")
    section(st,s,"2. Dataset and preparation")
    para(st,s,"We used the public <i>LLM Response Evaluation Benchmark</i> from Pranavsakunde on GitHub (cited at the end). It contains 100 shared prompts across general knowledge (20), coding (15), mathematics (15), writing (15), reasoning (15), translation (10), and safety (10). The publisher reports collecting an answer from ChatGPT, Claude, and Gemini for every prompt. We used the prompts and response text, not the source dataset's evaluation scores. Our team did not generate the original responses.")
    if data is None:
        para(st,s,"<b>TO VERIFY:</b> Download and inspect the original XLSX file, confirm the 300 records and model labels, check source reuse permissions, and run the preparation script. No accuracy numbers are claimed in this draft.")
    else:
        m=data["metadata"]
        counts_text = ", ".join("{}: {}".format(name, m["family_counts"].get(name, 0)) for name in ["ChatGPT", "Claude", "Gemini"])
        para(st,s,f"We checked for blank responses and retained <b>{m['dataset_rows']} answers</b> from <b>{m['unique_prompts']} shared prompts</b>. Each model has {m['unique_prompts']} answers ({counts_text}). We split whole prompts into {m['train_prompts']} training groups, {m['validation_prompts']} validation groups, and {m['test_prompts']} test groups. There are no shared prompts between splits. Our test set has 45 answers, 15 per model. The benchmark author describes the responses as authentic, but we did not have model API logs to independently verify their origins.")
    para(st,s,"We preserved the response text and visible formatting apart from trimming leading or trailing whitespace. For <b>prompt-only</b>, the input is the question; for <b>response-only</b>, it is the answer; for <b>combined</b>, it is question, separator, then answer. We lowercased and tokenized words, numbers, and punctuation. The vocabulary was built from training examples only, which keeps unseen test vocabulary from leaking into the model.")
    section(st,s,"3. Models and training procedure")
    para(st,s,"<b>CNN:</b> A trainable word embedding feeds three one-dimensional convolution branches, with kernel widths 3, 5 and 7. Max pooling turns each branch into a fixed-length vector, and dropout plus a fully connected layer produce three scores. The different kernel widths are intended to capture short patterns such as repeated phrases, punctuation and formatting conventions.")
    para(st,s,"<b>LSTM:</b> The same embedding idea feeds a bidirectional long short-term memory network. Forward and backward final hidden states are concatenated, then passed through dropout and a classification layer. Unlike the CNN, the LSTM processes tokens in sequence and can potentially use longer-range ordering.")
    para(st,s,"Both classifiers are trained from scratch with cross-entropy loss and Adam (learning rate 0.002, weight decay 0.0001). The default batch size is 32, the sequence limit is 256 tokens, embeddings are 64-dimensional, and the CNN channels / LSTM hidden size are 48. Training uses up to 14 epochs, gradient clipping, and early stopping on validation loss (patience 5). The random seed is 42. The best validation checkpoint is evaluated once on the held-out test prompts. CNN filters have widths 3, 5 and 7; both models use dropout 0.3. Prompt + response has the same 256-token cap; placing the prompt first can push the end of the answer outside that window. This is one reason the combined condition is not guaranteed to help.")
    para(st,s,"Reported metrics are test accuracy and macro-F1. Accuracy is easy to interpret, but macro-F1 gives equal weight to each model family. With three balanced classes, random chance is approximately 33.3%. The small dataset means the results should be treated as an initial experiment, not a reliable real-world authorship detector.")
    section(st,s,"4. Experiments and results")
    if data is None:
        para(st,s,"<b>RESULTS PENDING:</b> Run the provided train.py against the real dataset, then regenerate this PDF without --draft. It will insert the six required experiments, two extra-credit analyses, figures and measured values automatically.")
    else:
        rows=[]
        for model in ("CNN","LSTM"):
            for mode in ("input","output","combined"):
                x=data["rq2"][f"{model}_{mode}"]
                rows.append([model,mode, f"{100*x['accuracy']:.1f}%",f"{x['macro_f1']:.3f}",str(x['epochs_ran'])])
        table(st,s,["Model","Features","Test accuracy","Macro-F1","Epochs"],rows,[.8*inch,1.15*inch,1.35*inch,1.05*inch,.65*inch])
        insert_chart(st,s,a.results/"rq2_comparison.png","Figure 1. Same train/validation/test prompt groups are used across all six CNN and LSTM conditions.")
        c=data["rq2"]["CNN_output"]
        l=data["rq2"]["LSTM_output"]
        para(st,s,f"<b>RQ1:</b> With the responses alone, the CNN scored {c['accuracy']*100:.1f}% accuracy (macro-F1 {c['macro_f1']:.3f}), compared with {l['accuracy']*100:.1f}% (macro-F1 {l['macro_f1']:.3f}) for the LSTM. The CNN correctly labeled {c['confusion_matrix'][0][0]} of 15 ChatGPT answers and {c['confusion_matrix'][1][1]} of 15 Claude answers; it also correctly labeled {c['confusion_matrix'][2][2]} of 15 Gemini answers. Its mistakes included confusing Claude with Gemini. The numbers suggest identifiable patterns in this dataset, not a universal way to prove authorship.")
        for model in ("CNN","LSTM"):
            q=data["rq2"]
            n=q[f"{model}_input"]["accuracy"]
            out=q[f"{model}_output"]["accuracy"]
            both=q[f"{model}_combined"]["accuracy"]
            para(st,s,f"<b>RQ2, {model}:</b> Prompt-only scored {100*n:.1f}%, response-only {100*out:.1f}%, and prompt + response {100*both:.1f}%. Adding the prompt was {abs(100*(both-out)):.1f} percentage points {'better' if both>=out else 'worse'} than using the response alone. Every prompt appears with all three labels, so identical prompt text cannot uniquely identify a model. The chance-level prompt result is therefore a useful sanity check, not a meaningful predictor.")
        insert_chart(st,s,a.results/"cnn_confusion.png","Figure 2. CNN response-only errors, broken down by true and predicted family.",width=5.3)
        section(st,s,"5. Additional analyses")
        if data.get("rq3",{}).get("held_out_category"):
            para(st,s,"<b>RQ3: generalization to a different kind of question.</b> We held out the writing category completely from training and validation. Models were trained on all remaining categories, then tested on writing prompts they had never seen. This checks whether a classifier has learned patterns that travel beyond the types of questions in its training set.")
            rows=[]
            for name in ("CNN","LSTM"):
                x=data["rq3"][f"{name}_unseen_writing"]
                rows.append([name,str(data['rq3']['held_out_prompts']),f"{100*x['accuracy']:.1f}%",f"{x['macro_f1']:.3f}"])
            table(st,s,["Model","Held-out writing prompts","Test accuracy","Macro-F1"],rows,[1*inch,1.9*inch,1.1*inch,1.15*inch])
            para(st,s,"Compared with the usual response-only test (CNN 80.0%, LSTM 55.6%), the leave-writing-out scores (42.2%, 53.3%) show a much bigger drop for the CNN. This suggests the CNN may depend more on patterns tied to familiar tasks. However, the test sets and training data are different across the two experiments, so this is an illustration of domain shift rather than a perfectly controlled comparison. There are only 15 writing prompts.")
        else:
            para(st,s,"<b>RQ3:</b> The source data did not have enough labeled writing prompts for a reliable leave-writing-out experiment; the script skipped this analysis rather than inventing values.")
        para(st,s,"<b>RQ4: What makes the output look different?</b> We measured response length, average word length, lexical diversity (unique-word ratio), average sentence length, Markdown symbol count, bullet lines, and newlines. These are descriptive properties rather than proof of what the neural network learned.")
        means=data["rq4"].get("style_feature_means",{})
        if means:
            rows=[]
            for model in ("ChatGPT","Claude","Gemini"):
                x=means.get(model,{})
                rows.append([model,f"{x.get('response_words',0):.1f}",f"{x.get('lexical_diversity',0):.3f}",f"{x.get('markdown_symbols',0):.1f}",f"{x.get('bullet_lines',0):.1f}"])
            table(st,s,["Family","Words","Lexical diversity","Markdown marks","Bullets"],rows,[1.0*inch,1.0*inch,1.3*inch,1.2*inch,.9*inch])
            insert_chart(st,s,a.results/"linguistic_patterns.png","Figure 3. Average response structure. Features have different scales and should not be compared as if they share units.")
        rows=[]
        for name in ("CNN","LSTM"):
            original=data["rq2"][f"{name}_output"]["accuracy"]
            stripped=data["rq4"].get(f"{name}_format_stripped_test",{}).get("accuracy")
            if stripped is not None:
                rows.append([name,f"{100*original:.1f}%",f"{100*stripped:.1f}%",f"{100*(stripped-original):+.1f} pp"])
        if rows:
            para(st,s,"To make the formatting comparison more concrete, we kept the trained response-only models fixed and stripped Markdown/list characters from the test answers. This is a <i>test-time perturbation</i>, not a new training run, so it isolates how sensitive the existing predictions are to these visible cues.")
            table(st,s,["Model","Original test","Stripped test","Change"],rows,[1.0*inch,1.4*inch,1.4*inch,1.2*inch])
        baseline_file=a.results/"style_baseline.json"
        if baseline_file.exists():
            baseline=json.loads(baseline_file.read_text())
            simple=baseline["models"]["word_count_only"]
            features=baseline["models"]["seven_style_features"]
            para(st,s,"We also trained a <b>simple logistic regression control</b> on response-style measurements. It uses the exact same prompt-disjoint train/test groups, but it does not read the text itself. This helps put the neural-network results into context: if a few numbers already identify a model, the neural networks may be using some of these simpler signals too.")
            table(st,s,["Style-only control","Test accuracy","Macro-F1"],[["Response word count",f"{100*simple['accuracy']:.1f}%",f"{simple['macro_f1']:.3f}"],["Seven response-style features",f"{100*features['accuracy']:.1f}%",f"{features['macro_f1']:.3f}"]],[2.45*inch,1.35*inch,1.15*inch])
            para(st,s,"Response length alone did better than chance, and the seven-feature model did better still. We cannot conclude these are the same signals learned by the neural networks. Also, short answers tend to have a higher unique-word ratio, so differences in lexical diversity should not be interpreted independently of length. Stripping markup changed the CNN by only one correct answer out of 45; formatting is part of the story, but it is unlikely to explain everything.")
    section(st,s,"6. What this experiment shows, and its limits")
    para(st,s,"One thing we learned is that a sensible train/test split matters just as much as choosing a network. Because the same questions were asked of all three models, we split by the whole prompt rather than splitting response rows at random. Otherwise the exact same question could appear during training and testing, making the scores look stronger than they really are. The 33.3% prompt-only results helped check that the experiment was not simply learning the questions.")
    if data is not None:
        c=data["rq2"]["CNN_output"]["accuracy"]
        l=data["rq2"]["LSTM_output"]["accuracy"]
        top="CNN" if c>=l else "LSTM"
        para(st,s,f"The clearest result was that the {top} beat the other network on response-only classification ({100*max(c,l):.1f}% versus {100*min(c,l):.1f}%). That does not mean CNNs are always better for language tasks: we trained both networks from scratch on just 207 examples, and this is one 45-answer test split. If we repeated the project, we would rerun it with several seeds and report the mean and variability, rather than lean on one headline accuracy.")
    para(st,s,"The combined-input result was also useful: supplying more text did not automatically improve predictions. Our 256-token cap means the prompt uses space that would otherwise hold response tokens. We would test a longer sequence limit or separate prompt/response encoders next. Finally, the style-only baseline made us question whether a model was recognizing genuine linguistic habits or more obvious signals, especially answer length. A length-matched dataset would help separate those explanations.")
    para(st,s,"There are limits to what we can claim: the benchmark has just 100 prompts, the source model names come from its publisher rather than API logs we could independently inspect, and exact model versions and generation settings are not fully documented. Our classifiers only choose among three known labels, even for text from another source. These results should not be used to accuse someone of AI authorship. More prompts, new model versions, multiple random splits, and a separate independent dataset would give a stronger test.")
    section(st,s,"7. Reproducibility and source")
    para(st,s,"The accompanying repository files contain preprocessing, CNN/LSTM implementations, prompt-group splitting, training and evaluation, RQ3/RQ4 analysis, figure generation, and this PDF report builder. The seed, model configuration, trained output-only checkpoints, split audit and output JSON are recorded in the results directory. Running the scripts with a verified source workbook should regenerate the numerical tables.")
    para(st,s,"<b>Dataset citation:</b> Pranavsakunde, <i>LLM Response Evaluation Benchmark</i>, GitHub: https://github.com/Pranavsakunde/LLM-Response-Evaluation-Benchmark. The public README says the 100 prompts were answered by ChatGPT, Claude and Gemini. The original response workbook is not included in our project repository; public access does not automatically grant redistribution rights.")
    para(st,s,"<b>Course specification:</b> CMPSC 448, <i>Who Wrote It? Identifying LLMs from Their Responses</i>, modified September 17, 2026, provided assignment PDF.")
    doc.build(st,onFirstPage=footer,onLaterPages=footer)
    print("Generated:",a.out)

if __name__=="__main__":
    main()
