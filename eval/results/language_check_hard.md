# Everyday mistakes in all five languages, set 'hard' (typed text, the whole app)

Grammar model: openai/gpt-oss-120b

| Language | Input | Output | Correct |
|---|---|---|---|
| English | She don't like coffee. Her brother drink it every morning. | She doesn't like coffee. Her brother drinks it every morning. | yes |
| English | Yesterday we goes to the park and play cricket. It were a lot of fun. | Yesterday we went to the park and played cricket. It was a lot of fun. | yes |
| English | My friends is coming tomorrow. They has booked a hotel near the station. | My friends are coming tomorrow. They have booked a hotel near the station. | yes |
| English | He can speaks three languages, but he don't write any of them. | He can speak three languages, but he doesn't write any of them. | yes |
| Hindi | मेरी बहन कल बाजार गया था। उसने वहाँ से सब्जी खरीदा। | मेरी बहन कल बाजार गई थी। उसने वहाँ से सब्जी खरीदी। | yes |
| Hindi | मेरा भाई रोज़ सुबह दौड़ने जाती है। वह बहुत मेहनती लड़का हैं। | मेरा भाई रोज़ सुबह दौड़ने जाता है। वह बहुत मेहनती लड़का है। | yes |
| Hindi | बच्चे पार्क में खेल रहा है। उनकी माँ उन्हें बुला रहे हैं। | बच्चे पार्क में खेल रहे हैं। उनकी माँ उन्हें बुला रही है। | yes |
| Hindi | लड़कियाँ स्कूल जा रहा है। उनके पास किताबें नहीं था। | लड़कियाँ स्कूल जा रही हैं। उनके पास किताबें नहीं थीं। | yes |
| Marathi | माझा भाऊ काल मुंबईला गेली। तो तिथे दोन दिवस राहणार आहेत। | माझा भाऊ काल मुंबईला गेला। तो तिथे दोन दिवस राहणार आहे। | yes |
| Marathi | आम्ही रोज सकाळी फिरायला जाते। आमची आई आमच्यासाठी नाश्ता बनवतो। | आम्ही रोज सकाळी फिरायला जाणो। आमची आई आमच्यासाठी नाश्ता बनवते। | no (expected: आम्ही रोज सकाळी फिरायला जातो। आमची आई आमच्यासाठी नाश्ता बनवते।) |
| Marathi | मुलगी शाळेत गेला होता। तिने तिथे नवीन मैत्रीण बनवला। | मुलगी शाळेत गेली होती। तिने तिथे नवीन मैत्रीण बनवली। | yes |
| Marathi | ते लोक उद्या पुण्याला गेले। त्यांची गाडी सकाळी निघाली। | ते लोक उद्या पुण्याला जातील। त्यांची गाडी सकाळी निघेल। | yes |
| Bengali | তুমি কোথায় যাচ্ছে? আমি বাজারে যাচ্ছে। | তুমি কোথায় যাচ্ছ? আমি বাজারে যাচ্ছি। | yes |
| Bengali | আমরা প্রতিদিন সকালে চা খায়। আমার বাবা খবরের কাগজ পড়ি। | আমরা প্রতিদিন সকালে চা খাই। আমার বাবা খবরের কাগজ পড়ে। | yes |
| Bengali | সে আগামীকাল দিল্লি গেল। তার ভাই তার সঙ্গে গিয়েছিল। | সে আগামীকাল দিল্লি যাবে। তার ভাই তার সঙ্গে গিয়েছিল। | no (expected: সে আগামীকাল দিল্লি যাবে। তার ভাই তার সঙ্গে যাবে।) |
| Bengali | আমি গতকাল বাজারে যাই। সেখানে আমি মাছ কিনি। | আমি গতকাল বাজারে গেলাম। সেখানে আমি মাছ কিনলাম। | yes |
| Punjabi | ਮੇਰੀ ਭੈਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਿਆ ਸੀ। ਉਹ ਉੱਥੇ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲਿਆ। | ਮੇਰੀ ਭੈਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਈ ਸੀ। ਉਹ ਉੱਥੇ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲੀ। | yes |
| Punjabi | ਮੁੰਡੇ ਮੈਦਾਨ ਵਿੱਚ ਖੇਡ ਰਿਹਾ ਹੈ। ਕੁੜੀਆਂ ਘਰ ਵਿੱਚ ਪੜ੍ਹ ਰਹੇ ਹਨ। | ਮੁੰਡੇ ਮੈਦਾਨ ਵਿੱਚ ਖੇਡ ਰਹੇ ਹਨ। ਕੁੜੀਆਂ ਘਰ ਵਿੱਚ ਪੜ੍ਹ ਰਹੀਆਂ ਹਨ। | yes |
| Punjabi | ਅਸੀਂ ਹਰ ਰੋਜ਼ ਗੁਰਦੁਆਰੇ ਜਾਂਦਾ ਹਾਂ। ਮੇਰੀ ਮਾਂ ਉੱਥੇ ਸੇਵਾ ਕਰਦਾ ਹੈ। | ਅਸੀਂ ਹਰ ਰੋਜ਼ ਗੁਰਦੁਆਰੇ ਜਾਂਦੇ ਹਾਂ। ਮੇਰੀ ਮਾਂ ਉੱਥੇ ਸੇਵਾ ਕਰਦੀ ਹੈ। | yes |
| Punjabi | ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧਾ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤਾ। | ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧੀ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤੀ। | yes |

| Language | Corrected as expected |
|---|---|
| English | 4/4 |
| Hindi | 4/4 |
| Marathi | 3/4 |
| Bengali | 3/4 |
| Punjabi | 4/4 |
