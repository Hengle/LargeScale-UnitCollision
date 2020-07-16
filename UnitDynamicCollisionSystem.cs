using Unit;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;

public class UnitDynamicCollisionSystem : SystemBase
{
    // We separate the world into a 2d grid. This way we can check collisions between entities in the same cell and nearby cells only
    // IMPORTANT
    // The way my grid is structured is like this: first item is the number of entities in the current cell, and the next 4 are INDICES into a temp array
    // created every frame (YOU WILL SEE HOW THIS WORKS LATER)
    private NativeArray<int> grid; 
    // We also need a list of "active cells" which is cleared and refilled every frame, this is for optimization reasons as you will see
    private NativeList<int> activeCells;

    // Simple a ref to an entity query
    private EntityQuery unitQuery;

    // The bucket size is how many entities are allowed to be in on cell at a time. More allowed will mean more collisions checks, at reduced performance
    // Realistically, if you set the cell size to 1x1 meter, you wont ever have more than 4 people standing in that area in an rts, and if they are,
    // it will be too crowded to see. 
    // The cell size is the dimensions of the cell in world units, and the cells across is the length of one side of your map
    private const int BUCKET_SIZE = 4;
    private const float CELL_SIZE = 1f;
    private int CELLS_ACROSS = 1;

    protected unsafe override void OnStartRunning()
    {
        base.OnCreate();
        // For my game, soldiers have a "unit tag"
        unitQuery = GetEntityQuery( typeof( UnitTag ) );
        // I have a static signleton called "Game Handler" which contains another class called PathfindingGraph, which is made in the editor
        // The point is, I have some way to reference the game map, and derive the number of cells from it and my stated cell size
        CELLS_ACROSS = ( int ) ( ( GameHandler.instance.pathfindingGraph.cellSize * GameHandler.instance.pathfindingGraph.numCellsAcross ) / CELL_SIZE );
        // Store a persistent grid. This is our spatial division grid
        // It is bucketsze + 1 because we need to account for the special aray structure
        grid = new NativeArray<int>( CELLS_ACROSS * CELLS_ACROSS * ( BUCKET_SIZE + 1 ) , Allocator.Persistent );
        UnsafeUtility.MemSet( grid.GetUnsafePtr() , 0 , grid.Length );

        // Set an empty list the size of the number of units in game
        activeCells = new NativeList<int>( unitQuery.CalculateEntityCount() , Allocator.Persistent );
        activeCells.Clear();
    }
    protected override void OnUpdate()
    {
        unitQuery = GetEntityQuery( typeof( UnitTag ) ); // Must update every frame as units may die or spawn, in my game they only die
        // THIS IS THE ARRAY WHICH THE GRID REFERENCES
        // This array is perfectly aligned with the unit query
        // This means anytime we use a job with unitQuery, the array of entities is aligned with this temp array,
        // because of how we will fill this array
        // The spatial grid values reference an index into this array
        NativeArray<float2> copyPositions = new NativeArray<float2>( unitQuery.CalculateEntityCount() , Allocator.TempJob , NativeArrayOptions.ClearMemory );

        // These three jobs all run on a separate threed, but are single threded on that thread
        // This is faster than spreading each job over multiple threads
        //
        // So each frame
        // We must loop through all units and copy their positions to the temp array
        JobHandle copyJob = new CopyPositionsJob
        {
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            copyPositions = copyPositions ,
        }.ScheduleSingle( unitQuery , Dependency );
        // At the same time, we must build the map on a separate thread
        // This means we are looping through all units and updating the grid
        JobHandle updateMapJob = new BuildMapJob
        {
            BUCKET_SIZE = BUCKET_SIZE ,
            N_CELLS_ACROSS = CELLS_ACROSS ,
            CELL_SIZE = CELL_SIZE ,
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            grid = grid ,
        }.ScheduleSingle( unitQuery , Dependency );
        // At the same time, we must record which cells have changed
        // This is an optimization, as you will see
        // Basically, we are checking each unit the same was as in updateMapJob, but we are updating a separate, small list
        JobHandle recordActiveCellsJob = new RecordActiveBucketsJob
        {
            BUCKET_SIZE = BUCKET_SIZE ,
            nCellsAcross = CELLS_ACROSS ,
            cellSize = CELL_SIZE ,
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            activeBuckets = activeCells ,
        }.ScheduleSingle( unitQuery , Dependency );

        // All these jobs must be completed in order for the next to run, so here is a sync point
        JobHandle barrier = JobHandle.CombineDependencies( updateMapJob , recordActiveCellsJob );
        barrier = JobHandle.CombineDependencies( barrier , copyJob );
        barrier.Complete();

        // Here 
        JobHandle resolveCollisionsJob = new ResolveCollisionsJob
        {
            BUCKET_SIZE = BUCKET_SIZE ,
            CELL_SIZE = CELL_SIZE ,
            N_CELLS_ACROSS = CELLS_ACROSS ,
            copyPositions = copyPositions ,
            grid = grid ,
        }.Schedule( copyPositions.Length , 128 , barrier );
        JobHandle clearGridCountersJob = new ClearGridCountersJob
        {
            activeCells = activeCells ,
            grid = grid ,
        }.Schedule( resolveCollisionsJob );
        JobHandle clearChangesJob = new ClearChangesListJob
        {
            list = activeCells ,
        }.Schedule( clearGridCountersJob );
        JobHandle writeToUnitsJob = new WriteToUnitsJob
        {
            N_CELLS_ACROSS = CELLS_ACROSS ,
            CELL_SIZE = CELL_SIZE ,
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            copyPositions = copyPositions ,
        }.Schedule( unitQuery , resolveCollisionsJob );

        JobHandle disposeHandle = copyPositions.Dispose( writeToUnitsJob );
        JobHandle final = JobHandle.CombineDependencies( disposeHandle , clearChangesJob );
        Dependency = final;
    }
    protected override void OnDestroy()
    {
        if ( grid.IsCreated )
            grid.Dispose();
        if ( activeCells.IsCreated )
            activeCells.Dispose();
        base.OnDestroy();
    }

    //////////////
    //// JOBS ///
    ////////////

    [BurstCompile] private struct BuildMapJob : IJobChunk
    {
        [ReadOnly] public int BUCKET_SIZE;
        [ReadOnly] public int N_CELLS_ACROSS;
        [ReadOnly] public float CELL_SIZE;
        // We get this from feeding in the unit query on job declaration
        [ReadOnly] public ArchetypeChunkComponentType<Translation> archetypeTranslation;
        public NativeArray<int> grid;

        public void Execute( ArchetypeChunk chunk , int chunkIndex , int firstEntityIndex )
        {
            NativeArray<Translation> chunkTranslation = chunk.GetNativeArray( archetypeTranslation );

            // IMPORTANT
            // firstEntityIndex + i will point to the same location in the copy positions array created every frame
            for ( int i = 0; i < chunk.Count; i++ )
            {
                float px = chunkTranslation[ i ].Value.x;
                float py = chunkTranslation[ i ].Value.z;

                // The hash is just the cell index the unit is in
                int hash = ( int ) ( math.floor( px / CELL_SIZE ) + math.floor( ( py / CELL_SIZE ) ) * N_CELLS_ACROSS )
                int gridIndex = hash * BUCKET_SIZE;

                int count = grid[ gridIndex ];
                int cellIndex = gridIndex + 1;

                // Remember, the grid array is structured like (index 0 is the unit count of the cell, and the next 4 indices are 
                // the array indice of the temp copy positions array)
                // So we can put the entity index into the array based on the current cell's unit count
                // This is a really good optimization, but is very specific to my game, may not work for you
                if ( count < BUCKET_SIZE - 1 )
                {
                    grid[ cellIndex + count ] = firstEntityIndex + i;
                    grid[ gridIndex ] = count + 1;
                }
            }
        }
    }
    // This job is literally just recopying the poition data of units into an aligned array so we can efficiently use it with the grid 
    [BurstCompile] private struct CopyPositionsJob : IJobChunk
    {
        [ReadOnly] public ArchetypeChunkComponentType<Translation> archetypeTranslation;
        public NativeArray<float2> copyPositions;

        public void Execute( ArchetypeChunk chunk , int chunkIndex , int firstEntityIndex )
        {
            NativeArray<Translation> chunkTranslation = chunk.GetNativeArray( archetypeTranslation );

            for ( int i = 0; i < chunk.Count; i++ )
            {
                copyPositions[ firstEntityIndex + i ] = new float2(
                    chunkTranslation[ i ].Value.x , chunkTranslation[ i ].Value.z );
            }
        }
    }
    // This job's purpose is solely for cleanup
    // If you map is large, clearing the grid array every frame takes very long
    // Simply recording the active cells, and then clearing just those can be 100x + faster!
    // The reason this is a separate job from build map job is just so we can do it on another thread,
    // Even tho it seems like we are doing the same work twice, its actually faster as its split into 2 threads
    [BurstCompile] private struct RecordActiveBucketsJob : IJobChunk
    {
        [ReadOnly] public int BUCKET_SIZE;
        [ReadOnly] public int nCellsAcross;
        [ReadOnly] public float cellSize;
        [ReadOnly] public ArchetypeChunkComponentType<Translation> archetypeTranslation;

        public NativeList<int> activeBuckets;

        public void Execute( ArchetypeChunk chunk , int chunkIndex , int firstEntityIndex )
        {
            NativeArray<Translation> chunkTranslation = chunk.GetNativeArray( archetypeTranslation );

            // As you can see, we are basically doing the same thing as in build map job
            for ( int i = 0; i < chunk.Count; i++ )
            {
                float px = chunkTranslation[ i ].Value.x;
                float py = chunkTranslation[ i ].Value.z;

                int hash = ( int ) ( math.floor( px / cellSize ) + math.floor( ( py / cellSize ) ) * nCellsAcross );
                activeBuckets.Add( hash * BUCKET_SIZE );
            }
        }
    }

    // This jobs is fastest when using parallelfor
    [BurstCompile] private struct ResolveCollisionsJob : IJobParallelFor
    {
        [ReadOnly] public int BUCKET_SIZE;
        [ReadOnly] public float CELL_SIZE;
        [ReadOnly] public int N_CELLS_ACROSS;
        [ReadOnly] public NativeArray<int> grid;
        [NativeDisableParallelForRestriction] public NativeArray<float2> copyPositions;

        // This job loops over copypositions
        // This is much faster than looping over grid, as copypositions array is usually much smaller, unless your game is weird lol
        // Basically, in parallelfor, execute is run x times, x is determined by you. It cun loop over one array in parallel over many threads
        // So in this case, we loop over copypositions
        // REMEMBER THE ENITTY INDEX IS IMPLICITLY DEFINED BY COPYPOSITIONS INDEX, BECAUSE WE HAVE STRUCTURED IT TO BE ALIGNED WITH THE ENTITY QUERY
        public void Execute( int index )
        {
            float2 copyPosition = copyPositions[ index ];
            float px = copyPosition.x;
            float py = copyPosition.y;
            float radius = 0.25f;

            // Adjustement will be used later, this is the amount we need to displace the unit if it has collided
            float adjustmentX = px;
            float adjustmentY = py;

            int hash = ( int ) ( math.floor( px / CELL_SIZE ) + math.floor( py / CELL_SIZE ) * N_CELLS_ACROSS );
            int xR = ( int ) math.round( px );
            int yR = ( int ) math.round( py );
            int xD = math.select( 1 , -1 , xR < px );
            int yD = math.select( 1 , -1 , yR < py );

            // Basically we find the cell the position, and then check which other cells it is near
            // This math saves us from having to check all 8 surrounding cells to just 4
            // FixedList is a stack allocated array, ver very fast and very useful here
            FixedList32<int> hashes = new FixedList32<int>();
            hashes.Add( hash );

            bool xOffset = math.abs( xR - px ) < 0.3f;
            bool yOffset = math.abs( yR - py ) < 0.3f;

            if ( xOffset )
                hashes.Add( hash + xD );
            if ( yOffset )
                hashes.Add( hash + yD * N_CELLS_ACROSS );
            if ( xOffset && yOffset )
                hashes.Add( hash + yD * N_CELLS_ACROSS );

            // Now for each cell it is in or close to, we check the positions of nearby units
            // WE CAN DO THIS BECAUSE THE SPATIAL PARTITION GRID HOLDS THE INDICES OF COPYPOSITIONS
            for ( int i = 0; i < hashes.Length; i++ )
            {
                int gridIndex = hashes[ i ] * BUCKET_SIZE;
                int count = grid[ gridIndex ];
                int cellIndex = gridIndex + 1;

                for ( int j = 1; j < grid[ gridIndex ]; j++ )
                {
                    float2 testPosition = copyPositions[ grid[ cellIndex + j ] ];
                    float px2 = testPosition.x;
                    float py2 = testPosition.y;
                    float radius2 = 0.25f;

                    float distance = math.sqrt( ( px - px2 ) * ( px - px2 ) + ( py - py2 ) * ( py - py2 ) );
                    int overlaps = math.select( 0 , 1 , distance < radius + radius2 );

                    float overlap = 0.4f * ( distance - radius - radius2 );

                    adjustmentX -= overlaps * ( overlap * ( px - px2 ) ) / ( distance + 0.01f );
                    adjustmentY -= overlaps * ( overlap * ( py - py2 ) ) / ( distance + 0.01f );
                }
            }

            // Now we update the position of the unit int the array
            // This also means that if another unit has collided with this one, we have already displaced it,
            // another optimization
            copyPositions[ index ] = new float2( adjustmentX , adjustmentY );
        }
    }
    [BurstCompile] private struct WriteToUnitsJob : IJobChunk
    {
        [ReadOnly] public float CELL_SIZE;
        [ReadOnly] public int N_CELLS_ACROSS;
        [ReadOnly] public NativeArray<float2> copyPositions;
        public ArchetypeChunkComponentType<Translation> archetypeTranslation;

        public void Execute( ArchetypeChunk chunk , int chunkIndex , int firstEntityIndex )
        {
            NativeArray<Translation> chunkTranslation = chunk.GetNativeArray( archetypeTranslation );

            // AGAIN, SINCE COPY POSITIONS IS ALIGNED WITH UNIT QUERY, WRITING THE UPDATED POSITION DATA TO THE ENTITIES
            // IS EXTREMELY FAST, AND CAN TAKE ADVANTAGE OF SIMD (single instruction mulitple data)
            for ( int i = 0; i < chunk.Count; i++ )
            {
                float2 copyPosition = copyPositions[ firstEntityIndex + i ];

                chunkTranslation[ i ] = new Translation
                {
                    Value = new float3( copyPosition.x , chunkTranslation[ i ].Value.y , copyPosition.y ) ,
                };
            }
        }
    }

    // Cleanup
    [BurstCompile] private struct ClearGridCountersJob : IJob
    {
        [ReadOnly] public NativeArray<int> activeCells;
        public NativeArray<int> grid;

        public void Execute()
        {
            // Unrolled loop because UNITY doesn't automatically generate simd for this code for some reason
            // But see here, how in order to clear the grid we only need to loop over the much smaller activeCells array
            // Because we did the tiny amount of extra work earlier
            int i = 0;
            for ( ; i < activeCells.Length - 4; i += 4 )
            {
                int4 indices = activeCells.ReinterpretLoad<int4>( i );
                grid[ indices.x ] = 0;
                grid[ indices.y ] = 0;
                grid[ indices.z ] = 0;
                grid[ indices.w ] = 0;
            }
            for ( ; i < activeCells.Length; i++ )
            {
                grid[ activeCells[ i ] ] = 0;
            }
        }
    }
    [BurstCompile] private struct ClearChangesListJob : IJob
    {
        public NativeList<int> list;

        // EZ
        public void Execute()
        {
            list.Clear();
        }
    }
}