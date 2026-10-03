import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


app.registerExtension({
    name: "design61.FolderFramesVideoPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {

        if (
            nodeData.name !==
            "FolderFramesToVideoFFmpeg_design61"
        ) {
            return;
        }

        const oldOnNodeCreated =
            nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated =
            function () {

                const result =
                    oldOnNodeCreated?.apply(
                        this,
                        arguments
                    );

                const node = this;

                // --------------------------------------------
                // 创建 video DOM
                // --------------------------------------------

                const container =
                    document.createElement("div");

                container.style.width = "100%";
                container.style.paddingTop = "6px";
                container.style.boxSizing = "border-box";
                container.style.display = "none";


                const video =
                    document.createElement("video");

                video.controls = true;
                video.loop = true;
                video.muted = false;
                video.playsInline = true;

                video.style.width = "100%";
                video.style.maxHeight = "360px";
                video.style.objectFit = "contain";
                video.style.background = "#111";
                video.style.borderRadius = "6px";


                container.appendChild(video);


                const widget =
                    node.addDOMWidget(
                        "video_preview",
                        "video",
                        container,
                        {
                            serialize: false,
                            hideOnZoom: false,
                        }
                    );


                // --------------------------------------------
                // 防止 widget 自动绘制文本
                // --------------------------------------------

                widget.computeSize = function(width) {

                    if (
                        container.style.display ===
                        "none"
                    ) {
                        return [
                            width,
                            0
                        ];
                    }

                    return [
                        width,
                        300
                    ];
                };


                node._folderVideoElement =
                    video;

                node._folderVideoContainer =
                    container;

                return result;
            };


        const oldOnExecuted =
            nodeType.prototype.onExecuted;


        nodeType.prototype.onExecuted =
            function (message) {

                oldOnExecuted?.apply(
                    this,
                    arguments
                );


                const videos =
                    message?.videos;


                if (
                    !videos
                    ||
                    videos.length === 0
                ) {
                    return;
                }


                const info =
                    videos[0];


                const params =
                    new URLSearchParams({
                        filename:
                            info.filename,

                        type:
                            info.type || "output",

                        subfolder:
                            info.subfolder || "",
                    });


                const url =
                    api.apiURL(
                        `/view?${params.toString()}`
                    );


                const video =
                    this._folderVideoElement;

                const container =
                    this._folderVideoContainer;


                if (
                    !video
                    ||
                    !container
                ) {
                    return;
                }


                // 防止浏览器缓存旧视频
                video.src =
                    `${url}&t=${Date.now()}`;


                container.style.display =
                    "block";


                // 节点自动增高
                const minWidth =
                    Math.max(
                        this.size[0],
                        440
                    );


                this.setSize([
                    minWidth,
                    Math.max(
                        this.size[1],
                        720
                    )
                ]);


                app.graph.setDirtyCanvas(
                    true,
                    true
                );
            };
    }
});