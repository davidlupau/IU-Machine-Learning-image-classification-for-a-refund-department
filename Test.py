def extract_combined_features(image_path):
    """
    Extract shape, texture, edge, and color features from a single image.

    Parameters:
    image_path (str): Path to the image file

    Returns:
    dict: Dictionary containing all extracted features, or None if processing fails
    """
    try:
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((60, 80))
        img_array = np.array(image)
        gray_image = np.mean(img_array, axis=2)

        # Extract shape features
        shape_features = {
            'aspect_ratio': img_array.shape[0] / img_array.shape[1],
            'vertical_symmetry': np.mean(np.abs(gray_image - np.flipud(gray_image))),
            'horizontal_symmetry': np.mean(np.abs(gray_image - np.fliplr(gray_image)))
        }

        # Extract texture features
        lbp = feature.local_binary_pattern(gray_image, P=8, R=1)
        texture_features = {
            'texture_mean': lbp.mean(),
            'texture_var': lbp.var(),
            'texture_uniformity': len(np.unique(lbp)) / len(lbp.flatten())
        }

        # Extract edge features
        sobel_h = ndimage.sobel(gray_image, axis=0)
        sobel_v = ndimage.sobel(gray_image, axis=1)
        edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        canny_edges = feature.canny(gray_image, sigma=1.0)

        edge_features = {
            'edge_density': np.mean(edge_magnitude),
            'edge_variance': np.var(edge_magnitude),
            'horizontal_edges': np.mean(np.abs(sobel_h)),
            'vertical_edges': np.mean(np.abs(sobel_v)),
            'canny_edge_density': np.mean(canny_edges)
        }

        # Extract color features
        color_features = {}
        for idx, channel in enumerate(['red', 'green', 'blue']):
            channel_data = img_array[:,:,idx]
            color_features.update({
                f'mean_{channel}': channel_data.mean(),
                f'std_{channel}': channel_data.std(),
                f'skew_{channel}': stats.skew(channel_data.flatten())
            })

        # Add color ratios
        color_features.update({
            'red_green_ratio': color_features['mean_red'] / (color_features['mean_green'] + 1e-6),
            'blue_green_ratio': color_features['mean_blue'] / (color_features['mean_green'] + 1e-6),
            'color_variance': np.var([color_features['mean_red'],
                                    color_features['mean_green'],
                                    color_features['mean_blue']])
        })

        # Combine all features
        return {**shape_features, **texture_features, **edge_features, **color_features}

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None