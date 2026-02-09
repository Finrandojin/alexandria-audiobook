Quite a lot going on here, so here's the highlights:

* `index.html` is now modular and loaded at runtime to reduce the risk of git merge conflicts.
* JavaScript now separate from HTML; may split further on a functionality basis for reasons stated above.
* Editor tab allows merging and splitting of chunks. (Need to document functionality and do a couple of minor bugfixes.) New `insert-chunk` and `delete-chunk` endpoints to support the functionality.
* "Generate annotated script" is disabled until you've loaded something. (Should probably also do the same for "Review script"?)
* A few extra sanity checks here and there.
* Very basic "Readable script" functionality; will possibly be deprecated with further changes to the Editor.