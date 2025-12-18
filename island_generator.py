from PIL import Image, ImageDraw
import math
import random

TERRAIN_COLORS = {
    0: (0, 0, 0),      # unknown/void
    1: (34, 139, 34),  # grass
}

def generate_grid_system(grid_width=1000, grid_height=1000):
    """Generates a grid system for the game map."""
    grid = [[1 for _ in range(grid_width)] for _ in range(grid_height)]
    return grid

def _hash_noise(x, y, seed=0):
    """Deterministic pseudo-noise in [0, 1]."""
    return (math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453) % 1.0

def _lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t

def _smoothstep(t):
    """Smooth interpolation curve."""
    return t * t * (3.0 - 2.0 * t)

def _perlin_noise(x, y, seed=0):
    """Smooth Perlin-like noise in [0, 1] with bilinear interpolation."""
    # Get integer coordinates
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Get fractional part
    fx = x - x0
    fy = y - y0
    
    # Smooth the interpolation
    sx = _smoothstep(fx)
    sy = _smoothstep(fy)
    
    # Get corner values
    n00 = _hash_noise(x0, y0, seed)
    n10 = _hash_noise(x1, y0, seed)
    n01 = _hash_noise(x0, y1, seed)
    n11 = _hash_noise(x1, y1, seed)
    
    # Bilinear interpolation
    nx0 = _lerp(n00, n10, sx)
    nx1 = _lerp(n01, n11, sx)
    return _lerp(nx0, nx1, sy)

def zoomed_island_terrain(grid, zoom=0.03, octaves=6, water_level=0.40, stone_level=0.60, seed=None, min_score=0.3):
    """Zooms into a random region of Perlin noise to focus on island features. Returns (terrain_grid, noise_grid)."""
    h = len(grid)
    w = len(grid[0])
    
    # Random offset to pick a random location in the infinite noise space
    if seed is None:
        seed = random.randint(0, 10000)
    
    # Try random locations until we find one good enough
    offset_x, offset_y = 0, 0
    max_attempts = 50  # Safety limit to prevent infinite loop
    
    for attempt in range(max_attempts):
        test_x = random.randint(0, 100000)
        test_y = random.randint(0, 100000)
        
        # Sample a few points near center to find interesting regions
        center_samples = []
        for dy in [-10, 0, 10]:
            for dx in [-10, 0, 10]:
                sx = (w//2 + dx + test_x) * zoom
                sy = (h//2 + dy + test_y) * zoom
                
                val = 0.0
                amp = 1.0
                freq = 1.0
                max_val = 0.0
                
                for octave in range(octaves):
                    n = _perlin_noise(sx * freq, sy * freq, seed=seed + octave)
                    val += n * amp
                    max_val += amp
                    amp *= 0.5
                    freq *= 2.0
                
                center_samples.append(val / max_val)
        
        # Calculate variance - we want moderate variance (interesting terrain)
        avg = sum(center_samples) / len(center_samples)
        variance = sum((s - avg) ** 2 for s in center_samples) / len(center_samples)
        
        # Check if this region is good enough
        center_score = 1.0 - abs(avg - 0.5) * 2.0
        variance_score = min(variance * 10, 1.0)
        total_score = center_score * variance_score
        
        if total_score >= min_score:
            offset_x = test_x
            offset_y = test_y
            break
    
    # Generate noise values with offset
    noise_grid = [[0.0 for _ in range(w)] for _ in range(h)]
    
    for y in range(h):
        for x in range(w):
            # Sample from offset position (zoomed into random area)
            sample_x = (x + offset_x) * zoom
            sample_y = (y + offset_y) * zoom
            
            # Multi-octave Perlin noise
            value = 0.0
            amplitude = 1.0
            frequency = 1.0
            max_value = 0.0
            
            for octave in range(octaves):
                n = _perlin_noise(sample_x * frequency, sample_y * frequency, seed=seed + octave)
                value += n * amplitude
                max_value += amplitude
                amplitude *= 0.5
                frequency *= 2.0
            
            value /= max_value
            noise_grid[y][x] = value
            
            # Assign terrain based on noise value
            # Low and high elevations are grass islands, middle values are void
            if value < water_level:
                grid[y][x] = 1  # grass (low elevation islands)
            elif value < stone_level:
                grid[y][x] = 0  # unknown/void (empty space)
            else:
                grid[y][x] = 1  # grass (high elevation islands)
    
    return grid, noise_grid

def remove_edge_islands(grid, land_value=1, void_value=0):
    """Removes any landmasses that touch the edge of the grid using BFS."""
    from collections import deque
    h = len(grid)
    w = len(grid[0])
    visited = [[False for _ in range(w)] for _ in range(h)]
    
    def bfs_remove(start_x, start_y):
        """BFS to find and remove all tiles connected to an edge tile."""
        queue = deque([(start_x, start_y)])
        visited[start_y][start_x] = True
        to_remove = [(start_x, start_y)]
        
        while queue:
            x, y = queue.popleft()
            
            # Check 4 cardinal directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if not visited[ny][nx] and grid[ny][nx] == land_value:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
                        to_remove.append((nx, ny))
        
        # Remove all tiles in this connected component
        for rx, ry in to_remove:
            grid[ry][rx] = void_value
    
    # Check all edge tiles
    edge_tiles = []
    
    # Top and bottom edges
    for x in range(w):
        if grid[0][x] == land_value and not visited[0][x]:
            edge_tiles.append((x, 0))
        if grid[h-1][x] == land_value and not visited[h-1][x]:
            edge_tiles.append((x, h-1))
    
    # Left and right edges
    for y in range(h):
        if grid[y][0] == land_value and not visited[y][0]:
            edge_tiles.append((0, y))
        if grid[y][w-1] == land_value and not visited[y][w-1]:
            edge_tiles.append((w-1, y))
    
    # Remove all edge-touching islands
    for x, y in edge_tiles:
        if not visited[y][x] and grid[y][x] == land_value:
            bfs_remove(x, y)
    
    return grid

def find_largest_landmass(grid, target_value=1):
    """Uses BFS to find all connected components of target_value and returns the largest one as a set of (x, y) coordinates."""
    h = len(grid)
    w = len(grid[0])
    visited = [[False for _ in range(w)] for _ in range(h)]
    
    def bfs(start_x, start_y):
        """BFS to find all tiles connected to start position."""
        from collections import deque
        queue = deque([(start_x, start_y)])
        visited[start_y][start_x] = True
        component = {(start_x, start_y)}
        
        while queue:
            x, y = queue.popleft()
            
            # Check 4 cardinal directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if not visited[ny][nx] and grid[ny][nx] == target_value:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
                        component.add((nx, ny))
        
        return component
    
    # Find all connected components
    components = []
    for y in range(h):
        for x in range(w):
            if grid[y][x] == target_value and not visited[y][x]:
                component = bfs(x, y)
                components.append(component)
    
    # Return largest component
    if components:
        return max(components, key=len)
    return set()

def extract_and_resize_island(grid, landmass_coords, target_width, target_height, fill_value=0, land_value=1):
    """Extracts a landmass and resizes it to fill the target grid dimensions."""
    if not landmass_coords:
        return [[fill_value for _ in range(target_width)] for _ in range(target_height)]
    
    # Find bounding box of the landmass
    xs = [x for x, y in landmass_coords]
    ys = [y for x, y in landmass_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    island_width = max_x - min_x + 1
    island_height = max_y - min_y + 1
    
    # Calculate scale to fit island to grid with some padding
    padding = 20  # pixels of padding from edge
    scale_x = (target_width - 2 * padding) / island_width
    scale_y = (target_height - 2 * padding) / island_height
    scale = min(scale_x, scale_y)  # Use uniform scale to preserve shape
    
    # Calculate new dimensions and centering offset
    new_width = int(island_width * scale)
    new_height = int(island_height * scale)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    
    # Create new grid
    new_grid = [[fill_value for _ in range(target_width)] for _ in range(target_height)]
    
    # Map each target pixel back to source coordinates
    for ty in range(target_height):
        for tx in range(target_width):
            # Calculate source coordinates
            sx = int((tx - offset_x) / scale + min_x)
            sy = int((ty - offset_y) / scale + min_y)
            
            # Check if this maps to a landmass tile
            if (sx, sy) in landmass_coords:
                new_grid[ty][tx] = land_value
    
    return new_grid

def generate_grid_visual(grid):
    """Generates a visual representation of the grid using PIL."""
    WIDTH = len(grid[0])
    HEIGHT = len(grid)
    PX_PER_TILE = 10

    img = Image.new("RGB", (WIDTH * PX_PER_TILE, HEIGHT * PX_PER_TILE))
    draw = ImageDraw.Draw(img)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            draw.rectangle(
                [x * PX_PER_TILE, y * PX_PER_TILE, (x + 1) * PX_PER_TILE, (y + 1) * PX_PER_TILE],
                outline="black",
                fill=TERRAIN_COLORS.get(grid[y][x], (0, 0, 0))
            )

    img.show()
    img.save("example_map_gen_imgs/grid_visual.png")

def main():
    GRID_WIDTH = 500
    GRID_HEIGHT = 500

    # Generate initial terrain with multiple islands
    grid = generate_grid_system(GRID_WIDTH, GRID_HEIGHT)
    island_terrain, island_noise = zoomed_island_terrain(grid, zoom=0.03, octaves=6)
    
    print("Displaying initial terrain with multiple islands...")
    generate_grid_visual(island_terrain)
    
    # Remove any islands touching the edges
    print("Removing edge-touching islands...")
    remove_edge_islands(island_terrain, land_value=1, void_value=0)
    
    print("Displaying terrain after removing edge islands...")
    generate_grid_visual(island_terrain)
    
    # Find largest landmass using BFS
    print("Finding largest landmass...")
    largest_landmass = find_largest_landmass(island_terrain, target_value=1)
    print(f"Found landmass with {len(largest_landmass)} tiles")
    
    # Extract and resize the landmass to fill the grid
    print("Extracting and resizing island to fill grid...")
    final_grid = extract_and_resize_island(island_terrain, largest_landmass, GRID_WIDTH, GRID_HEIGHT)
    
    print("Displaying final resized island...")
    generate_grid_visual(final_grid)

if __name__ == "__main__":
    main()
