# SereTod Motivation
Task-oriented dialogue (TOD) systems are designed to assist users to accomplish their goals, and have gained
more and more attention recently in both academia and
industry with the current advances in neural approaches
[1]. A TOD system typically consists of several modules,
which track user goals to update dialog states, query a
task-related knowledge base (KB) using the dialog states,
decide actions and generate responses. Unfortunately,
building TOD systems remains a label-intensive, timeconsuming task for two main reasons. First, training
neural TOD systems requires manually labeled dialog
states and system acts (if used), in both traditional
modular approach [12, 8] and recent end-to-end trainable
approach [11, 7, 4, 9, 13]. Second, it is often assumed
that a task-related knowledge base is available. But for
system development from scratch in many real-world
tasks, expert labors are needed to construct the KB from
annotating unstructured data. Thus, the labeled-data
scarcity challenge hinders efficient development of TOD
systems at scale.
