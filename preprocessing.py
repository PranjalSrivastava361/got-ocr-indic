 import os
 import json
 import pandas as pd
 # Load your CSV file (change the filename if needed)
 csv_path = '/kaggle/input/hindi-dataset/data_80k/data.csv'
  # Replace with actual CSV filename
 df = pd.read_csv(csv_path)
 image_base_path = '/kaggle/input/hindi-dataset/data_80k/output_images/'
 json_data = []
 for index, row in df.iterrows():
    full_image_path = os.path.join(image_base_path, row['image_file'])
    json_obj = {
        "query": "<image>Transcribe the text in this image",
        "response": row['text'],
        "images": [full_image_path]
    }
    json_data.append(json_obj)
 # Save to JSON
 output_path = '/kaggle/working/output_data.json'
 with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)
 print(f"JSON saved to: {output_path}")
 # Example - 
# 'images': ['/kaggle/input/hindi-dataset/data_80k/TestSamples/2.png']},
 # {'query': '<image>Transcribe the text in this image',
 # 'response': 'नज़रया: गोरखपुर, नागपुर और िद ी के ि कोण म फंसा है 2019',
 JSON saved to: /kaggle/working/output_data.jso
