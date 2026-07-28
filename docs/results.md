# Results

The footage covers four cameras for about 2 to 2.5 minutes each, around 8:10 PM on 10 April 2026. The bills cover the whole day: 24 bills made of 101 line items. Every figure below comes from the saved outputs of the pipeline; amounts in rupees stay out of this public repo.

## Sales (POS, whole day)

| Finding | Value |
| --- | --- |
| Busiest two hours | 6 PM to 8 PM: 8 of 24 bills (33%) and 38% of revenue |
| Revenue concentration | The top 20% of bills (5 of 24) bring in 56% of revenue |
| Basket size | Mean basket is 1.7x the median |

All three come from SQL in `src/metrics.py`: a self-join on hourly buckets for the peak, and window functions for the top 20% and the median. With 24 bills from one day, treat them as a description of that day rather than a pattern.

## Shelf attention

Checked against people counted every 2 seconds on the shelf camera (63 moments):

| | Lakme wall | Summer display | Ratio |
| --- | --- | --- | --- |
| Labelled frames, person-seconds | 214 | 106 | 2.0x |
| Pipeline, fine-tuned detector | 158 | 98 | 1.6x |
| Pipeline, original detector | 158 | 110 | 1.4x |
| Mean error per moment (people) | 0.63 | 0.48 | |

The pipeline gets the direction right and understates the gap. When three people stand at the Lakme wall, the summer display in front hides their feet, so some of them get counted at the display.

## Door

Ground truth: nobody entered or left during the clip, while 20 to 25 people walked past outside and one man stood at the door on his phone.

| Door line | False entries and exits |
| --- | --- |
| As configured | 0 |
| Stretched across the whole frame | 36 |
| Drawn across the street | 33 |
| Across the street, without the noise guards | 55 |

## Detector

On held-out frames (the last 30% of each clip):

| Model | Precision | Recall | mAP50 | mAP50-95 | CPU ms per frame |
| --- | --- | --- | --- | --- | --- |
| yolo11n, pretrained | 0.92 | 0.69 | 0.81 | 0.59 | about 90 |
| yolo11n, fine-tuned on 328 store frames | 0.93 | 0.87 | 0.94 | 0.77 | about 90 |
| yolo11x (the teacher) | 0.97 | 0.89 | 0.97 | 0.91 | about 830 |

The teacher's score flatters it, because the labels started as its own boxes. On the entrance camera the fine-tuned model's precision dropped from 0.79 to 0.59: most of the "wrong" boxes are real people behind the glass that the labels missed, and the labels were left alone rather than corrected with the model's own output.

## Staff or customer

103 labelled tracks, 43 of them staff:

| Method | Precision | Recall | F1 | PR-AUC | Customers kept, in-store |
| --- | --- | --- | --- | --- | --- |
| Zero-shot CLIP | 0.73 | 1.00 | 0.84 | 0.84 | 13% |
| Linear probe on CLIP embeddings | 0.82 | 0.98 | 0.89 | 0.96 | 40% |
| Counter rule (billing camera only) | 1.00 | 0.86 | 0.92 | n/a | n/a |

The probe is the best of the three, but in-store it still calls most customers staff. The in-store customer examples come from only two people, and they dress much like the uniform. So customer-only numbers are a lower bound, and the report always shows everyone alongside.

## Store metrics for the filmed window

| Metric | Value |
| --- | --- |
| Entries through the door | 0 |
| Billing queue | at most 2 people, all staff working the counter |
| Conversion | undefined: nobody entered, and the nearest bills are at 7:54 PM and 8:25 PM, outside the filmed window |
