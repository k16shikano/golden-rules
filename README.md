原稿と編集後のセクションのペアから「どんな編集をしたか」の傾向を抽出し、それを新規文章の修正案で利用する

-  `make_golden_rule.py` ：「どんな編集をしたか」の傾向をGPT-4で分類
-  `refine_script.py` ：「どんな編集をしたか」の傾向のうち、新規文章との距離がSentenceTransformerの意味で近いものを選ぶ（`find_top_patterns`）。それを「お手本」として埋め込んだプロンプトで新規入力の文章をGPT-4に書き換えさせる（`refine_text_with_gpt4`）
-  `sse_server.py` ：Cursorなどから接続して `refine_script` を呼び出すMCPサーバー
