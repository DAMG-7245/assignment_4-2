import os
import json
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

# 确保 API 密钥被正确地设置和读取
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

client = Mistral(api_key=api_key)

try:
    # 调用 OCR API，并打印响应来进行调试
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2201.04234"
        },
        include_image_base64=True
    )
    print("OCR Response类型:", type(ocr_response))
    
    # 转换响应为字典格式
    response_dict = ocr_response.model_dump()
    
    # 检查响应结构
    if "pages" in response_dict:
        pages = response_dict["pages"]
        print(f"找到 {len(pages)} 页")
        
        # 创建一个与官方示例相同格式的输出
        result = {"pages": []}
        
        for page in pages:
            page_data = {
                "index": page.get("index", 0)
            }
            
            # 如果有markdown，添加到结果中
            if "markdown" in page:
                page_data["markdown"] = page["markdown"]
            
            result["pages"].append(page_data)
        
        # 输出格式化的JSON结果
        print("\n提取的内容:")
        print(json.dumps(result, indent=2))
        
        # 另外单独提取每页的markdown内容
        print("\n各页面内容:")
        for i, page in enumerate(result["pages"]):
            if "markdown" in page:
                print(f"\n--- 第 {page['index']} 页 ---")
                print(page["markdown"])
            else:
                print(f"\n--- 第 {page['index']} 页 ---")
                print("此页无markdown内容")
    
    # 如果上面的方法失败，尝试直接从对象中获取
    elif hasattr(ocr_response, "pages"):
        pages = ocr_response.pages
        print(f"找到 {len(pages)} 页对象")
        
        result = {"pages": []}
        
        for page in pages:
            page_data = {
                "index": page.index if hasattr(page, "index") else 0
            }
            
            if hasattr(page, "markdown"):
                page_data["markdown"] = page.markdown
            
            result["pages"].append(page_data)
        
        # 输出格式化的JSON结果
        print("\n提取的内容:")
        print(json.dumps(result, indent=2))
        
        # 单独提取markdown内容
        print("\n各页面内容:")
        for i, page in enumerate(result["pages"]):
            if "markdown" in page:
                print(f"\n--- 第 {page['index']} 页 ---")
                print(page["markdown"])
            else:
                print(f"\n--- 第 {page['index']} 页 ---")
                print("此页无markdown内容")
    
    else:
        print("未能从响应中找到pages")
        print("响应键:", response_dict.keys())

except Exception as e:
    print(f"处理OCR请求时出错: {e}")
    import traceback
    traceback.print_exc()