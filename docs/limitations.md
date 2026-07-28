# Limitations and next steps

## What limits the numbers today

- **Very little footage.** About 2.5 minutes per camera from one evening. Nobody came through the door, and no bill falls inside the filmed window. That's why door recall and conversion can't be measured yet, even though the code for both exists.
- **One ID per camera.** Each camera numbers people on its own, and someone hidden for a moment can come back with a new ID. The funnel is therefore a set of stage counts, not one person followed from door to till, and many shelf visits are cut short.
- **Customers in black.** The staff classifier keeps only about 40% of in-store customers. The customer examples come from two people, so customer-only numbers are a lower bound.
- **Small labelled samples.** 63 shelf-occupancy moments (each count could be off by one), 103 labelled tracks and 24 bills. All of it is directional, not statistically tight.
- **Labels from the same process.** The detector's validation boxes were drafted by the same teacher model as the training boxes, so the fine-tuned model is partly graded on matching that teacher's style.
- **One blurred clock.** The entrance camera's burned-in clock is unreadable, so its start time is estimated to within about 30 seconds. That only affects time-based joins on that camera.

## What I'd do next

1. Get a full trading day of footage that overlaps the bills, so conversion and door recall become measurable.
2. Re-identify people across cameras with appearance embeddings, and stitch broken tracks. That turns the funnel into real journeys and cuts the number of cut-off visits.
3. Hand-label a small validation set independently of the teacher model.
4. Map each camera to the floor plan (a homography), so zones live in floor coordinates and hidden feet can be estimated from the head.
5. Label more in-store customers, from several days, and group the cross-validation folds by person rather than by track.
6. Put bootstrap intervals on the sales shares and the shelf ratio before quoting them.
7. At 40 stores: batch the video work per store and day on a GPU, move from SQLite to Postgres, and keep the API as a thin read layer.
