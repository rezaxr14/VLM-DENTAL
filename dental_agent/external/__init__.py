"""External (hosted) model evaluation through the project's own agent loop, tools, harness and clinical evaluator.

Nothing in this package is imported by ``dental_agent.{agent,model,tools,evaluation,harness}``; it lives outside those
folders on purpose, because ``NotebookEvaluations._source_identity`` hashes every ``.py`` there and a new file would
change the identity of every existing result bundle.
"""
