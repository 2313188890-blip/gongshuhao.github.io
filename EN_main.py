from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化FastAPI
app = FastAPI(title="模型展示Demo")
# 配置模板目录（用于渲染前端页面）
templates = Jinja2Templates(directory="templates")

# 加载模型和分词器（替换为你的模型路径）
model_path = "F:\output"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 模型设为评估模式
model.eval()
# 若有GPU可使用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# 定义预测函数（核心：调用模型处理输入）
def predict(text: str) -> str:
    """对输入文本进行预测（以情感分类为例：0=负面，1=正面）"""
    # 文本预处理
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 模型预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=1).item()  # 取概率最大的类别

    # 转换标签为可读结果（根据你的模型调整）
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    return label_map[pred_label]


# 首页路由（返回交互页面）
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",  # 渲染的模板
        {"request": request, "result": None}  # 传递给模板的参数（初始无结果）
    )


# 预测接口（接收前端表单提交，返回结果页面）
@app.post("/predict", response_class=HTMLResponse)
def predict_endpoint(request: Request, text: str = Form(...)):
    # 调用预测函数
    result = predict(text)
    # 渲染页面并返回结果
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "input_text": text}
    )


@app.post("/predict_chinese", response_class=HTMLResponse)
def predict_endpoint(request: Request, text: str = Form(...)):
    # 调用预测函数
    result = predict(text)
    # 渲染页面并返回结果
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "input_text": text}
    )
