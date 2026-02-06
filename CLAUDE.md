Concise, explanatory, professional code, comment and documentation style
utilizing comprehensive logging modules for precise debugging
When encountering bugs or errors in the code you wrote, just try to fix it
concisely record bugs and solutions in tip.md.

File structure should resemble that of each phase and pipeline.
Script execution should never touch existing database to prevent data lost.
All script should write and utilize comprehensive and concise logging modules at src/logging/

Do not provide default behaviors or use placeholders for any parameters or function values. Instead, if one isn't provided or passed in, raise error and log.
Pipelines that can be applied in parallel using multi-thread or multi-worker should implement simple concurrent coding practices.
Do not provide overcomplicated solution and coding styles for a feature or bug that could be implemented straightforward.
Use dataset/sample-test-50.tar for all testing purposes as it contains 50 random samples from the marginalia dump.