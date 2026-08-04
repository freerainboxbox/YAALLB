## Yet Another Apple LLM Load Balancer

A VRAM-aware LLM load balancer, quick and dirty, to point to your existing LLM runners, and is VRAM-aware.

Currently targets LM Studio and antirez/ds4. More may be supported later.

You can specify a VRAM limit on your Apple Silicon Mac in MB, and YAALLB will set `iogpu.wired_limit_mb` to that limit and respect your memory by unloading the least recently used model when asked.

TODO: Config JSON docs and CLI running docs