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
    private NativeArray<int> grid;
    private NativeList<int> activeCells;

    private EntityQuery unitQuery;

    private const int BUCKET_SIZE = 4;
    private const float CELL_SIZE = 1f;
    private int CELLS_ACROSS = 1;

    protected unsafe override void OnStartRunning()
    {
        base.OnCreate();
        unitQuery = GetEntityQuery( typeof( UnitTag ) );
        CELLS_ACROSS = ( int ) ( ( GameHandler.instance.pathfindingGraph.cellSize * GameHandler.instance.pathfindingGraph.numCellsAcross ) / CELL_SIZE );
        grid = new NativeArray<int>( CELLS_ACROSS * CELLS_ACROSS * ( BUCKET_SIZE + 1 ) , Allocator.Persistent );
        UnsafeUtility.MemSet( grid.GetUnsafePtr() , 0 , grid.Length );

        activeCells = new NativeList<int>( unitQuery.CalculateEntityCount() , Allocator.Persistent );
        activeCells.Clear();
    }
    protected override void OnUpdate()
    {
        unitQuery = GetEntityQuery( typeof( UnitTag ) );
        NativeArray<float2> copyPositions = new NativeArray<float2>( unitQuery.CalculateEntityCount() , Allocator.TempJob , NativeArrayOptions.ClearMemory );

        JobHandle copyJob = new CopyPositionsJob
        {
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            copyPositions = copyPositions ,
        }.ScheduleSingle( unitQuery , Dependency );
        JobHandle updateMapJob = new BuildMapJob
        {
            BUCKET_SIZE = BUCKET_SIZE ,
            N_CELLS_ACROSS = CELLS_ACROSS ,
            CELL_SIZE = CELL_SIZE ,
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            grid = grid ,
        }.ScheduleSingle( unitQuery , Dependency );
        JobHandle recordActiveCellsJob = new RecordActiveBucketsJob
        {
            BUCKET_SIZE = BUCKET_SIZE ,
            nCellsAcross = CELLS_ACROSS ,
            cellSize = CELL_SIZE ,
            archetypeTranslation = GetArchetypeChunkComponentType<Translation>() ,
            activeBuckets = activeCells ,
        }.ScheduleSingle( unitQuery , Dependency );

        JobHandle barrier = JobHandle.CombineDependencies( updateMapJob , recordActiveCellsJob );
        barrier = JobHandle.CombineDependencies( barrier , copyJob );
        barrier.Complete();

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

    // Build
    [BurstCompile] private struct BuildMapJob : IJobChunk
    {
        [ReadOnly] public int BUCKET_SIZE;
        [ReadOnly] public int N_CELLS_ACROSS;
        [ReadOnly] public float CELL_SIZE;
        [ReadOnly] public ArchetypeChunkComponentType<Translation> archetypeTranslation;
        public NativeArray<int> grid;

        public void Execute( ArchetypeChunk chunk , int chunkIndex , int firstEntityIndex )
        {
            NativeArray<Translation> chunkTranslation = chunk.GetNativeArray( archetypeTranslation );

            for ( int i = 0; i < chunk.Count; i++ )
            {
                float px = chunkTranslation[ i ].Value.x;
                float py = chunkTranslation[ i ].Value.z;

                int hash = ( int ) ( math.floor( px / CELL_SIZE ) + math.floor( ( py / CELL_SIZE ) ) * N_CELLS_ACROSS );
                int gridIndex = hash * BUCKET_SIZE;

                int count = grid[ gridIndex ];
                int cellIndex = gridIndex + 1;

                if ( count < BUCKET_SIZE - 1 )
                {
                    grid[ cellIndex + count ] = firstEntityIndex + i;
                    grid[ gridIndex ] = count + 1;
                }
            }
        }
    }
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

            for ( int i = 0; i < chunk.Count; i++ )
            {
                float px = chunkTranslation[ i ].Value.x;
                float py = chunkTranslation[ i ].Value.z;

                int hash = ( int ) ( math.floor( px / cellSize ) + math.floor( ( py / cellSize ) ) * nCellsAcross );
                activeBuckets.Add( hash * BUCKET_SIZE );
            }
        }
    }

    // Use
    [BurstCompile] private struct ResolveCollisionsJob : IJobParallelFor
    {
        [ReadOnly] public int BUCKET_SIZE;
        [ReadOnly] public float CELL_SIZE;
        [ReadOnly] public int N_CELLS_ACROSS;
        [ReadOnly] public NativeArray<int> grid;
        [NativeDisableParallelForRestriction] public NativeArray<float2> copyPositions;

        public void Execute( int index )
        {
            float2 copyPosition = copyPositions[ index ];
            float px = copyPosition.x;
            float py = copyPosition.y;
            float radius = 0.25f;

            float adjustmentX = px;
            float adjustmentY = py;

            int hash = ( int ) ( math.floor( px / CELL_SIZE ) + math.floor( py / CELL_SIZE ) * N_CELLS_ACROSS );
            int xR = ( int ) math.round( px );
            int yR = ( int ) math.round( py );
            int xD = math.select( 1 , -1 , xR < px );
            int yD = math.select( 1 , -1 , yR < py );

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

        public void Execute()
        {
            list.Clear();
        }
    }
}