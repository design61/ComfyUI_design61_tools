export function refreshV39ReferenceImagesForSampler(node) {
    const input = node.inputs?.find(item => item.name === "reference_images");
    const link = node.graph?.links?.[input?.link];
    const helper = node.graph?.getNodeById?.(link?.origin_id);
    if (helper?.comfyClass === "H3ContinuumReferenceImagesV39") helper.__h3ReferenceImagesRefresh?.();
}
