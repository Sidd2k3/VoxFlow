"""Everyday grammar mistakes in every supported language, with the expected correction.

Used by tests/test_languages.py (rules) and eval/check_languages.py (the whole app, with the real LLM).
"""

CASES = {
    "en-US": [
        ("She doesn't knows what I was talking about.", "She doesn't know what I was talking about."),
        ("I seen it yesterday.", "I saw it yesterday."),
        ("She was looking at me like she knowed me.", "She was looking at me like she knew me."),
        ("He go to school every day.", "He goes to school every day."),
        ("I could of done it.", "I could have done it."),
    ],
    "hi-IN": [
        ("मैं कल बाजार गया है।", "मैं कल बाजार गया था।"),
        ("हम लोग दिल्ली में रहता है।", "हम लोग दिल्ली में रहते हैं।"),
        ("वह खाना खाया।", "उसने खाना खाया।"),
        ("मैं स्कूल जाता है।", "मैं स्कूल जाता हूँ।"),
        ("लड़की स्कूल जाता है।", "लड़की स्कूल जाती है।"),
        ("मेरी बहन कल आया था।", "मेरी बहन कल आई थी।"),
    ],
    "mr-IN": [
        ("आम्ही शाळेत जात आहे।", "आम्ही शाळेत जात आहोत।"),
        ("मी काल बाजारात जातो।", "मी काल बाजारात गेलो।"),
        ("मी शाळेत जातो आहे।", "मी शाळेत जात आहे।"),
        ("ती घरी जातो।", "ती घरी जाते।"),
        ("मुलगी शाळेत गेला।", "मुलगी शाळेत गेली।"),
    ],
    "bn-IN": [
        ("আমি গতকাল স্কুলে যাই।", "আমি গতকাল স্কুলে গেলাম।"),
        ("সে বাজারে যাই।", "সে বাজারে যায়।"),
        ("আমি ভাত খেয়েছে।", "আমি ভাত খেয়েছি।"),
        ("আমি বই পড়ে।", "আমি বই পড়ি।"),
        ("সে গতকাল বাড়ি যায়।", "সে গতকাল বাড়ি গেল।"),
    ],
    "pa-IN": [
        ("ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹੈ।", "ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ।"),
        ("ਮੈਂ ਕੱਲ੍ਹ ਬਾਜ਼ਾਰ ਗਿਆ ਹੈ।", "ਮੈਂ ਕੱਲ੍ਹ ਬਾਜ਼ਾਰ ਗਿਆ ਸੀ।"),
        ("ਉਹ ਰੋਟੀ ਖਾਧੀ।", "ਉਸਨੇ ਰੋਟੀ ਖਾਧੀ।"),
        ("ਅਸੀਂ ਘਰ ਜਾਂਦਾ ਹੈ।", "ਅਸੀਂ ਘਰ ਜਾਂਦੇ ਹਾਂ।"),
        ("ਕੁੜੀ ਸਕੂਲ ਜਾਂਦਾ ਹੈ।", "ਕੁੜੀ ਸਕੂਲ ਜਾਂਦੀ ਹੈ।"),
        ("ਮੇਰੀ ਬਹਨ ਕੱਲ ਬਾਜ਼ਾਰ ਗਿਆ ਹੈ।", "ਮੇਰੀ ਬਹਨ ਕੱਲ ਬਾਜ਼ਾਰ ਗਈ ਸੀ।"),   # live: two mistakes, ਭੈਣ heard as ਬਹਨ
    ],
}

# Other answers that are also correct (the LLM may legitimately choose them)
ALSO_CORRECT = {
    "वह खाना खाया।": {"उसने खाना खाया था।"},          # past perfect is also grammatical
}

# Realistic everyday sentences, one mistake each (the second live-test set). English, Hindi and Marathi
# also go through the LLM; for some English ones only the LLM can fix them.
REAL_CASES = {
    "en-US": [
        ("My manager don't want us to work on weekends.", "My manager doesn't want us to work on weekends."),
        ("Yesterday we goes to the new mall near the station.", "Yesterday we went to the new mall near the station."),
        ("She have been working here for three years.", "She has been working here for three years."),
    ],
    "hi-IN": [
        ("मेरी बहन अगले महीने दिल्ली जाएगा।", "मेरी बहन अगले महीने दिल्ली जाएगी।"),
        ("हम लोग कल रात फिल्म देखने गया था।", "हम लोग कल रात फिल्म देखने गए थे।"),
        ("मेरे पापा रोज सुबह पार्क में टहलती हैं।", "मेरे पापा रोज सुबह पार्क में टहलते हैं।"),
    ],
    "mr-IN": [
        ("माझी बहीण काल ऑफिसला उशिरा गेला।", "माझी बहीण काल ऑफिसला उशिरा गेली।"),
        ("आम्ही रविवारी क्रिकेट खेळायला जातो आहे।", "आम्ही रविवारी क्रिकेट खेळायला जात आहोत।"),
        ("तो रोज सकाळी व्यायाम करते।", "तो रोज सकाळी व्यायाम करतो।"),
    ],
    "bn-IN": [
        ("আমরা প্রতিদিন সকালে হাঁটতে যায়।", "আমরা প্রতিদিন সকালে হাঁটতে যাই।"),
        ("আমার ভাই গতকাল অফিসে যায়।", "আমার ভাই গতকাল অফিসে গেল।"),
        ("সে রোজ সকালে চা খাই।", "সে রোজ সকালে চা খায়।"),
    ],
    "pa-IN": [
        ("ਮੇਰਾ ਭਰਾ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਈ ਸੀ।", "ਮੇਰਾ ਭਰਾ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਿਆ ਸੀ।"),
        ("ਅਸੀਂ ਹਰ ਐਤਵਾਰ ਗੁਰਦੁਆਰੇ ਜਾਂਦਾ ਹਾਂ।", "ਅਸੀਂ ਹਰ ਐਤਵਾਰ ਗੁਰਦੁਆਰੇ ਜਾਂਦੇ ਹਾਂ।"),
        ("ਉਹ ਕੁੜੀ ਬਹੁਤ ਵਧੀਆ ਗਾਉਂਦਾ ਹੈ।", "ਉਹ ਕੁੜੀ ਬਹੁਤ ਵਧੀਆ ਗਾਉਂਦੀ ਹੈ।"),
    ],
}
ALSO_CORRECT.update({
    "She have been working here for three years.": {"She has worked here for three years."},
    "Yesterday we goes to the new mall near the station.": {"Yesterday, we went to the new mall near the station."},
})


# Two sentences with 2-3 mistakes each, as people actually speak: the case the user asked for
# ("2 or 3 sentences with many grammatical errors"). Listed alternatives are also correct.
HARD_CASES = {
    "en-US": [
        ("She don't like coffee. Her brother drink it every morning.",
         "She doesn't like coffee. Her brother drinks it every morning."),
        ("Yesterday we goes to the park and play cricket. It were a lot of fun.",
         "Yesterday we went to the park and played cricket. It was a lot of fun."),
        ("My friends is coming tomorrow. They has booked a hotel near the station.",
         "My friends are coming tomorrow. They have booked a hotel near the station."),
        ("He can speaks three languages, but he don't write any of them.",
         "He can speak three languages, but he doesn't write any of them."),
    ],
    "hi-IN": [
        ("मेरी बहन कल बाजार गया था। उसने वहाँ से सब्जी खरीदा।", "मेरी बहन कल बाजार गई थी। उसने वहाँ से सब्जी खरीदी।"),
        ("मेरा भाई रोज़ सुबह दौड़ने जाती है। वह बहुत मेहनती लड़का हैं।",
         "मेरा भाई रोज़ सुबह दौड़ने जाता है। वह बहुत मेहनती लड़का है।"),
        ("बच्चे पार्क में खेल रहा है। उनकी माँ उन्हें बुला रहे हैं।", "बच्चे पार्क में खेल रहे हैं। उनकी माँ उन्हें बुला रही है।"),
        ("लड़कियाँ स्कूल जा रहा है। उनके पास किताबें नहीं था।", "लड़कियाँ स्कूल जा रही हैं। उनके पास किताबें नहीं थीं।"),
    ],
    "mr-IN": [
        ("माझा भाऊ काल मुंबईला गेली। तो तिथे दोन दिवस राहणार आहेत।", "माझा भाऊ काल मुंबईला गेला। तो तिथे दोन दिवस राहणार आहे।"),
        ("आम्ही रोज सकाळी फिरायला जाते। आमची आई आमच्यासाठी नाश्ता बनवतो।",
         "आम्ही रोज सकाळी फिरायला जातो। आमची आई आमच्यासाठी नाश्ता बनवते।"),
        ("मुलगी शाळेत गेला होता। तिने तिथे नवीन मैत्रीण बनवला।", "मुलगी शाळेत गेली होती। तिने तिथे नवीन मैत्रीण बनवली।"),
        ("ते लोक उद्या पुण्याला गेले। त्यांची गाडी सकाळी निघाली।", "ते लोक उद्या पुण्याला जातील। त्यांची गाडी सकाळी निघेल।"),
    ],
    "bn-IN": [
        ("তুমি কোথায় যাচ্ছে? আমি বাজারে যাচ্ছে।", "তুমি কোথায় যাচ্ছ? আমি বাজারে যাচ্ছি।"),
        ("আমরা প্রতিদিন সকালে চা খায়। আমার বাবা খবরের কাগজ পড়ি।", "আমরা প্রতিদিন সকালে চা খাই। আমার বাবা খবরের কাগজ পড়েন।"),
        ("সে আগামীকাল দিল্লি গেল। তার ভাই তার সঙ্গে গিয়েছিল।", "সে আগামীকাল দিল্লি যাবে। তার ভাই তার সঙ্গে যাবে।"),
        ("আমি গতকাল বাজারে যাই। সেখানে আমি মাছ কিনি।", "আমি গতকাল বাজারে গিয়েছিলাম। সেখানে আমি মাছ কিনেছিলাম।"),
    ],
    "pa-IN": [
        ("ਮੇਰੀ ਭੈਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਿਆ ਸੀ। ਉਹ ਉੱਥੇ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲਿਆ।", "ਮੇਰੀ ਭੈਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਈ ਸੀ। ਉਹ ਉੱਥੇ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲੀ।"),
        ("ਮੁੰਡੇ ਮੈਦਾਨ ਵਿੱਚ ਖੇਡ ਰਿਹਾ ਹੈ। ਕੁੜੀਆਂ ਘਰ ਵਿੱਚ ਪੜ੍ਹ ਰਹੇ ਹਨ।", "ਮੁੰਡੇ ਮੈਦਾਨ ਵਿੱਚ ਖੇਡ ਰਹੇ ਹਨ। ਕੁੜੀਆਂ ਘਰ ਵਿੱਚ ਪੜ੍ਹ ਰਹੀਆਂ ਹਨ।"),
        ("ਅਸੀਂ ਹਰ ਰੋਜ਼ ਗੁਰਦੁਆਰੇ ਜਾਂਦਾ ਹਾਂ। ਮੇਰੀ ਮਾਂ ਉੱਥੇ ਸੇਵਾ ਕਰਦਾ ਹੈ।", "ਅਸੀਂ ਹਰ ਰੋਜ਼ ਗੁਰਦੁਆਰੇ ਜਾਂਦੇ ਹਾਂ। ਮੇਰੀ ਮਾਂ ਉੱਥੇ ਸੇਵਾ ਕਰਦੀ ਹੈ।"),
        ("ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧਾ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤਾ।", "ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧੀ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤੀ।"),
    ],
}
ALSO_CORRECT.update({
    "Yesterday we goes to the park and play cricket. It were a lot of fun.":
        {"Yesterday, we went to the park and played cricket. It was a lot of fun."},
    "बच्चे पार्क में खेल रहा है। उनकी माँ उन्हें बुला रहे हैं।": {"बच्चे पार्क में खेल रहे हैं। उनकी माँ उन्हें बुला रही हैं।"},
    "আমরা প্রতিদিন সকালে চা খায়। আমার বাবা খবরের কাগজ পড়ি।": {"আমরা প্রতিদিন সকালে চা খাই। আমার বাবা খবরের কাগজ পড়ে।"},
    "সে আগামীকাল দিল্লি গেল। তার ভাই তার সঙ্গে গিয়েছিল।": {"সে আগামীকাল দিল্লি যাবে। তার ভাই তার সঙ্গে যাচ্ছে।"},
    "তুমি কোথায় যাচ্ছে? আমি বাজারে যাচ্ছে।": {"তুমি কোথায় যাচ্ছো? আমি বাজারে যাচ্ছি।"},   # যাচ্ছো: common spelling
    "আমি গতকাল বাজারে যাই। সেখানে আমি মাছ কিনি।": {"আমি গতকাল বাজারে গেলাম। সেখানে আমি মাছ কিনলাম।",
                                                "আমি গতকাল বাজারে গিয়েছিলাম। সেখানে আমি মাছ কিনলাম।",
                                                "আমি গতকাল বাজারে গেলাম। সেখানে আমি মাছ কিনেছিলাম।"},
})
