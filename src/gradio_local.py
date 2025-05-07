import os
import gradio as gr
from style_transfer.inference import submit_function, person_example_fn



HEADER = """
<h1 style="text-align: center;"> Virtual-TryOn: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
"""

def app_gradio():
    with gr.Blocks(title="Virtual-TryOn") as demo:
        gr.Markdown(HEADER)

        with gr.Tabs():

            # --- Style Transfer Tab ---
            with gr.Tab("Style Transfer"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=350):
                        image_path = gr.Image(type="filepath", interactive=True, visible=False)
                        person_image = gr.Image(interactive=True, label="Person Image", type="filepath")
                        with gr.Row():
                            cloth_image = gr.Image(interactive=True, label="Condition Image", type="filepath")
                            cloth_type = gr.Radio(
                                label="Try-On Cloth Type",
                                choices=["upper", "lower", "overall"],
                                value="upper",
                            )
                        submit_style = gr.Button("Submit Style Transfer")

                        with gr.Accordion("Advanced Options", open=False):
                            num_inference_steps = gr.Slider(
                                label="Inference Step", minimum=10, maximum=100, step=5, value=50
                            )
                            guidance_scale = gr.Slider(
                                label="CFG Strength", minimum=0.0, maximum=50, step=0.5, value=2.5
                            )
                            seed = gr.Slider(
                                label="Seed", minimum=-1, maximum=10000, step=1, value=42
                            )

                    with gr.Column(scale=2, min_width=500):
                        with gr.Tabs():
                            with gr.Tab("Masked Image"):
                                mask_image = gr.Image(interactive=False, label="Masked Image")
                            with gr.Tab("Result Image"):
                                result_image = gr.Image(interactive=False, label="Result")

                        with gr.Row():
                            root_path = "style_transfer/resource/demo/example"
                            with gr.Column():
                                gr.Examples(
                                    examples=[os.path.join(root_path, "person", "men", f) for f in os.listdir(os.path.join(root_path, "person", "men"))],
                                    examples_per_page=4,
                                    inputs=image_path,
                                    label="Person Examples ①"
                                )
                                gr.Examples(
                                    examples=[os.path.join(root_path, "person", "women", f) for f in os.listdir(os.path.join(root_path, "person", "women"))],
                                    examples_per_page=4,
                                    inputs=image_path,
                                    label="Person Examples ②"
                                )
                                gr.Markdown(
                                    '<span style="color: #808080; font-size: small;">*Person examples from <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>.</span>'
                                )
                            with gr.Column():
                                gr.Examples(
                                    examples=[os.path.join(root_path, "condition", "upper", f) for f in os.listdir(os.path.join(root_path, "condition", "upper"))],
                                    examples_per_page=4,
                                    inputs=cloth_image,
                                    label="Condition Upper Examples"
                                )
                                gr.Examples(
                                    examples=[os.path.join(root_path, "condition", "overall", f) for f in os.listdir(os.path.join(root_path, "condition", "overall"))],
                                    examples_per_page=4,
                                    inputs=cloth_image,
                                    label="Condition Overall Examples"
                                )
                                gr.Examples(
                                    examples=[os.path.join(root_path, "condition", "person", f) for f in os.listdir(os.path.join(root_path, "condition", "person"))],
                                    examples_per_page=4,
                                    inputs=cloth_image,
                                    label="Condition Reference Person Examples"
                                )

                image_path.change(person_example_fn, inputs=image_path, outputs=person_image)
                submit_style.click(
                    fn=submit_function,
                    inputs=[
                        person_image,
                        cloth_image,
                        cloth_type,
                        num_inference_steps,
                        guidance_scale,
                        seed
                    ],
                    outputs=[mask_image, result_image],
                )

            # --- Pose Transfer Tab ---
            with gr.Tab("Pose Transfer"):
                with gr.Row():
                    with gr.Column():
                        person_pose_img = gr.Image(interactive=True, label="Person Image", type="filepath")
                        pose_target_img = gr.Image(interactive=True, label="Pose Image", type="filepath")
                        submit_pose = gr.Button("Submit Pose Transfer")

                        with gr.Accordion("Advanced Options", open=False):
                            steps_pose = gr.Slider(label="Inference Step", minimum=10, maximum=100, step=5, value=50)
                            scale_pose = gr.Slider(label="CFG Strength", minimum=0.0, maximum=50, step=0.5, value=2.5)
                            seed_pose = gr.Slider(label="Seed", minimum=-1, maximum=10000, step=1, value=42)

                    with gr.Column():
                        result_pose = gr.Image(interactive=False, label="Result")

                # Uncomment when you implement `pose_transfer_function`
                # submit_pose.click(
                #     fn=pose_transfer_function,
                #     inputs=[person_pose_img, pose_target_img, steps_pose, scale_pose, seed_pose],
                #     outputs=result_pose,
                # )

            # --- Style + Pose Transfer Tab ---
            with gr.Tab("Reel Generation"):
                with gr.Row():
                    with gr.Column():
                        person_mix_img = gr.Image(interactive=True, label="Person Image", type="filepath")
                        cloth_mix_img = gr.Image(interactive=True, label="Condition Image", type="filepath")
                        cloth_type_mix = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )
                        submit_mix = gr.Button("Submit Style + Pose Transfer")

                        with gr.Accordion("Advanced Options", open=False):
                            steps_mix = gr.Slider(label="Inference Step", minimum=10, maximum=100, step=5, value=50)
                            scale_mix = gr.Slider(label="CFG Strength", minimum=0.0, maximum=50, step=0.5, value=2.5)
                            seed_mix = gr.Slider(label="Seed", minimum=-1, maximum=10000, step=1, value=42)

                    with gr.Column():
                        result_mix = gr.Image(interactive=False, label="Result")

                # Uncomment when implemented
                # submit_mix.click(
                #     fn=style_and_pose_transfer_function,
                #     inputs=[person_mix_img, cloth_mix_img, cloth_type_mix, pose_mix_img, steps_mix, scale_mix, seed_mix],
                #     outputs=result_mix,
                # )

    demo.queue().launch(share=True, show_error=True)

if __name__ == "__main__":
    app_gradio()
