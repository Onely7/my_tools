# モデルパラメータ読み込み
state_dict = torch.load("./pytorch_model.bin")
if hasattr(model, "module"):
  model.module.load_state_dict(state_dict)
else:
  model.load_state_dict(state_dict)

checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
model = BertModel.from_pretrained(checkpoint)

# 予測
test_outputs, test_labels = [], []
# 学習は行わないため、学習にしか関係しない計算は省くことでコストを下げ速度を上げる
with torch.no_grad():
    for batch in test_dataloader:
        # 予測計算
        outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                token_type_ids=batch["token_type_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
        )
        outputs = outputs.cpu() # モデル結果がGPUに乗ったままになっているのでCPUに送信する。
        test_outputs.append(outputs)
        test_labels.append(batch["labels"])
            
    test_outputs = torch.cat(test_outputs, dim=0) # 出力を連結する
    test_labels = torch.cat(test_labels, dim=0) # 正答ラベルを連結する
            
    scores = calc_accuracy(test_outputs, test_labels) # 正答率で評価を行う
    print(f'テストデータに対する正答率: {scores:.4f}')
