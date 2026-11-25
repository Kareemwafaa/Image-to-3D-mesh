"""The image-to-mesh pipeline, one module per stage:

1. segmentation  - remove the background, centre the object (U^2-Net)
2. multiview     - hallucinate 6 consistent views (Zero123++ diffusion)
3. cameras       - the fixed camera poses those 6 views correspond to
4. neus          - neural SDF field + volume renderer (NeuS)
5. reconstruct   - fit the field to the views (multi-view optimisation)
6. meshing       - marching cubes on the SDF, colour, export .glb
"""
