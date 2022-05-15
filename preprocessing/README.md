# CICIDS2017 Dataset



## Data Pre-processing of the CICIDS2017

| N°    | Feature                            | Kept  | Modified | Dropped |    Note    |
| ----- |:----------------------------------:| -----:| --------:| -------:|-----------:|
| 0     | destination\_port                  |   x   |          |         |            |
| 1     | flow\_duration                     |       |          |    x    | Correlated |
| 2-3   | total\_fwd/bwd\_packet             |       |          |    x    | Correlated |
| 4-5   | total\_length\_of\_fwd/bwd\_packet |       |          |    x    | Correlated |
| 6-7   | fwd\_packet\_length\_max/mean      |       |          |    x    | Correlated |
| 8-9   | fwd\_packet\_length\_min/std       |   x   |          |         |            |
| 10-11 | bwd\_packet\_length\_max/mean      |       |          |    x    | Correlated |
| 12-13 | bwd\_packet\_length\_min/std       |   x   |          |         |            |
| 14    | flow\_bytes/s                      |       |     x    |         | NaN/Infinity|
| 15    | flow\_packet/s                     |       |     x    |         | NaN/Infinity|
| 16-19 | flow\_iat\_mean/std/max/min        |   x   |          |         |            |
| 20-24 | fwd\_iat\_total/mean/std/max/min   |   x   |          |         |            |
| 25-29 | bwd\_iat\_total/mean/std/max/min   |   x   |          |         |            |
| 30    | fwd\_psh\_flag                     |       |          |    x    | Correlated |
| 31    | bwd\_psh\_flag                     |       |          |    x    | No variance|
| 32    | fwd\_urg\_flag                     |       |          |    x    | No variance|
| 33    | bwd\_urg\_flag                     |       |          |    x    | No variance|
| 34-35 | fwd/bwd\_header\_length            |   x   |          |         |            |
| 36-37 | fwd/bwd\_packet/s                  |   x   |          |         |            |
| 38-39 | packet\_length\_max/mean           |   x   |          |         |            |
| 40-42 | packet\_length\_min/std/variance   |       |          |    x    | Correlated |
| 43    | fin\_flag\_count                   |   x   |          |         |            |
| 44    | syn\_flag\_count                   |   x   |          |         |            |
| 45    | rst\_flag\_count                   |       |          |    x    | Correlated |
| 46    | psh\_flag\_count                   |   x   |          |         |            |
| 47    | ack\_flag\_count                   |   x   |          |         |            |
| 48    | urg\_flag\_count                   |   x   |          |         |            |
| 49    | cwe\_flag\_count                   |       |          |    x    | No variance|
| 50    | ece\_flag\_count                   |   x   |          |         |            |
| 51    | down/up\_ratio                     |   x   |          |         |            |
| 52    | average\_packet\_size              |   x   |          |         |            |
| 53-54 | avg\_fwd/bwd\_segment size         |   x   |          |         |            |
| 55    | fwd\_header\_length.1              |       |          |    x    | Duplicate  |
| 56    | fwd\_avg\_bytes/bulk               |       |          |    x    | No variance|
| 57    | fwd\_avg\_packet/bulk              |       |          |    x    | No variance|
| 58    | fwd\_avg\_bulk rate                |       |          |    x    | No variance|
| 59    | bwd\_avg\_bytes/bulk               |       |          |    x    | No variance|
| 60    | bwd\_avg\_packet/bulk              |       |          |    x    | No variance|
| 61    | bwd\_avg\_bulk rate                |       |          |    x    | No variance|
| 62-63 | subflow\_fwd/bwd\_packets          |       |          |    x    | Correlated |
| 64-65 | subflow\_fwd/bwd\_bytes            |   x   |          |         |            |
| 66    | init\_win\_bytes\_forward          |   x   |          |         |            |
| 67    | init\_win\_bytes\_backward         |   x   |          |         |            |
| 68    | act\_data\_pkt\_fwd                |   x   |          |         |            |
| 69    | min\_seg\_size\_forward            |   x   |          |         |            |
| 70-73 | active\_mean/std/max/min           |   x   |          |         |            |
| 74-75 | idle\_mean/max                     |       |          |    x    | Correlated |
| 76-77 | idle\_std/min                      |   x   |          |         |            |
| 78    | label                              |       |          |    x    | Evaluation |