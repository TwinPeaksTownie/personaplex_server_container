# Cloud Deployment Roadmap: Two Paths

 You have two clear options for your "Launchable" build.

 ## Option A: "The Scrappy POC" (NVIDIA T4)
 **Goal:** Prove it runs on the absolute cheapest hardware.
 - **Hardware:** NVIDIA T4 (16GB VRAM)
 - **Cost:** ~$0.50 - $0.90 / hr
 - **Strategy:** **8-bit Quantization**.
   - You *must* compress the model to fit in 16GB.
   - Requires custom Dockerfile with `bitsandbytes`.
   - **Trade-off:** Slightly lower audio quality, higher engineering complexity.

 ### T4 Deployment Recipe
 1.  **Docker Base:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
 2.  **Key Libs:** `bitsandbytes`, `scipy`
 3.  **Command:** `python3 -m moshi.server --quantize 8bit ...`

 ---

 ## Option B: "The Powerhouse" (NVIDIA A40)
 **Goal:** Best quality, easiest setup, incredible value.
 - **Hardware:** NVIDIA A40 (48GB VRAM)
 - **Cost:** ~$0.64 / hr (CUDO Compute via Brev)
 - **Strategy:** **Full Precision (BF16)**.
   - 48GB is massive. You can run the full uncompressed model + the OS + overhead with 20GB to spare.
   - **No custom code needed.** Just standard Moshi install.
   - **Trade-off:** None. It's actually *simpler* than the T4 build.

 ### A40 Deployment Recipe
 1.  **Docker Base:** `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
 2.  **Key Libs:** Standard `moshi` pip install.
 3.  **Command:** `python3 -m moshi.server ...` (No special flags needed)

 ## Recommendation
 **Use the A40.**
 At **$0.64/hr**, it is cheaper than many T4 instances but offers **3x the VRAM (48GB vs 16GB)** and significantly newer architecture (Ampere vs Turing). It will run Laura at full fidelity with zero quantization artifacts.
