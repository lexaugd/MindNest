## Timestamp

Timestamps can be used in conjunction with andy server side implementation of
the interfaces. It can be useful to return information about when a 
particular resource has been created, updated, or accessed. To simplify the 
specification in the document we have not explicitly listed that a timestamp 
is part of the reource, but we can assume it may be added as part of the 
service implementation. To obtain an example timestamp a simple get function 
is provided. 


### Timestamp {#sec:spec-timestamp}

{include=./spec/timestamp.md}

#### timestamp.yaml


```{include=./spec/timestamp.yaml}
```

